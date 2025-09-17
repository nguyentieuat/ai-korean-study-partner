# %%writefile /content/train_solver_mcq_llama3.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train solver MCQ (A/B/C/D) cho TOPIK I (bước 3 — Solver) với Meta-Llama-3-8B-Instruct.

- Dữ liệu jsonl: mỗi dòng có keys:
  - instruction_ko (str)
  - context_ko (str)
  - choices (list[str])  # dạng ["A) ...", "B) ...", "C) ...", "D) ..."]
  - label (str)          # "A" | "B" | "C" | "D"

- Prompt kết thúc bằng TAG <ANS>. Chỉ tính loss sau TAG.
- QLoRA 4-bit (bitsandbytes) để vừa Colab T4 (16GB).
- Có --no-merge để chỉ lưu LoRA adapter (khuyến nghị cho POC trên Colab).
"""

import os, io, json, argparse, random, sys
from typing import Dict, Any, Iterable, List, Tuple

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel

# ====== Cấu hình TAG để mask loss sau TAG
ANSWER_TAG = "<ANS>"
TAG_VARIANTS = ["<ANS>", "\n<ANS>", " <ANS>"]  # chấp nhận vài biến thể để chắc chắn match


# --------------------------
# Utils
# --------------------------
def set_seed_all(seed: int = 13):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(p: str) -> Iterable[Dict[str, Any]]:
    with io.open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_prompt(ex: Dict[str, Any]) -> str:
    """Ghép prompt chuẩn: kết thúc bằng TAG; nhãn thật (A/B/C/D) nối sau."""
    inst, ctx = ex["instruction_ko"], ex["context_ko"]
    choices = "\n".join(ex["choices"])
    return (
        f"{inst}\n\n"
        f"지문:\n{ctx}\n\n"
        f"보기:\n{choices}\n\n"
        "정답은 A/B/C/D 중 하나의 글자만 쓰세요.\n"
        f"{ANSWER_TAG}"
    )


class ListJsonlDataset(TorchDataset):
    def __init__(self, path: str):
        self.rows: List[Dict[str, str]] = []
        for ex in read_jsonl(path):
            prompt = build_prompt(ex)
            # nối thêm 1 space trước nhãn cho dễ nhìn (không ảnh hưởng masking)
            self.rows.append({"text": prompt + " " + ex["label"]})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def find_any_subseq(hay: List[int], needles: List[List[int]]) -> Tuple[int, int]:
    """
    Tìm vị trí sớm nhất của BẤT KỲ biến thể TAG nào trong chuỗi token 'hay'.
    Trả về (pos_bat_dau, do_dai_mau) hoặc (-1, 0) nếu không thấy.
    """
    best = (len(hay) + 1, 0)  # (pos, L)
    for nd in needles:
        L = len(nd)
        if L == 0:
            continue
        for i in range(0, len(hay) - L + 1):
            if hay[i:i + L] == nd:
                if i < best[0]:
                    best = (i, L)
                break
    if best[0] <= len(hay):
        return best
    return (-1, 0)


def make_collate_fn(tok: AutoTokenizer, max_len: int):
    # Encode tất cả biến thể TAG (không thêm special tokens)
    template_variants = [tok.encode(v, add_special_tokens=False) for v in TAG_VARIANTS]
    template_variants = [ids for ids in template_variants if len(ids) > 0]
    assert len(template_variants) > 0, "Tokenizer không mã hoá được TAG; hãy đổi ANSWER_TAG/TAG_VARIANTS."

    def _get_text(item):
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            for k in ("text", "input", "prompt", "source"):
                if k in item:
                    return item[k]
        return str(item)

    def collate(batch):
        texts = [_get_text(b) for b in batch]
        enc = tok(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            add_special_tokens=True,
            return_tensors=None,
        )
        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]

        labels = []
        hits = 0
        for ids in input_ids:
            pos, L = find_any_subseq(ids, template_variants)
            if pos == -1:
                labels.append([-100] * len(ids))
            else:
                start = pos + L
                lab = [-100] * start + ids[start:]
                labels.append(lab)
                hits += 1

        if hits < len(input_ids):
            print(f"[WARN] mask: found {hits}/{len(input_ids)} TAGs in batch")

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def _pretty_bnb_error_and_exit(e: Exception):
    msg = (
        "\n[bitsandbytes ERROR]\n"
        "Có vẻ môi trường của bạn chưa sẵn sàng cho QLoRA 4-bit (bnb/triton).\n"
        "Thử chạy lần lượt các lệnh sau trong Colab, rồi chạy lại script:\n\n"
        "pip uninstall -y bitsandbytes\n"
        "pip install --no-cache-dir 'bitsandbytes==0.43.1' 'triton==2.3.0'\n\n"
        "Sau đó, nếu Colab đang tìm libbitsandbytes_cuda126.so thì copy từ 121:\n"
        "import os, site, shutil\n"
        "libdir=[p for p in site.getsitepackages() if p.endswith('dist-packages') or p.endswith('site-packages')][0]+'/bitsandbytes'\n"
        "shutil.copy(os.path.join(libdir,'libbitsandbytes_cuda121.so'), os.path.join(libdir,'libbitsandbytes_cuda126.so'))\n\n"
        f"Chi tiết lỗi gốc: {repr(e)}\n"
    )
    print(msg, file=sys.stderr)
    sys.exit(1)


# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out-adapter", required=True)
    ap.add_argument("--out-merged", required=True)
    ap.add_argument("--base", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bsz", type=int, default=2)           # T4: 4-bit, bsz=2 an toàn
    ap.add_argument("--grad-accum", type=int, default=32)   # batch hiệu dụng ≈ 64
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--eval-strategy", default="no")        # "steps" | "epoch" | "no"
    ap.add_argument("--cache-dir", default="/content/cache/hf")
    ap.add_argument("--no-merge", action="store_true", help="Chỉ lưu adapter; bỏ qua merge LoRA (khuyên dùng trên Colab).")
    args = ap.parse_args()

    os.makedirs(args.out_adapter, exist_ok=True)
    os.makedirs(args.out_merged, exist_ok=True)
    set_seed_all(args.seed)

    print("Loading datasets…")
    train_ds = ListJsonlDataset(args.train)
    val_ds = ListJsonlDataset(args.val)

    print("Loading tokenizer/model…")
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True, cache_dir=args.cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # QLoRA: load 4-bit để khớp T4 16GB
    try:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,  # T4: fp16
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base,
            cache_dir=args.cache_dir,
            quantization_config=bnb,
            device_map="auto",
        )
    except Exception as e:
        _pretty_bnb_error_and_exit(e)
        return  # never reached

    # LoRA cho Llama-3
    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_cfg)

    # Cấu hình training-friendly
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Sanity: tỉ lệ giữ được TAG sau tokenize+truncate
    tmpl_vars = [tok.encode(v, add_special_tokens=False) for v in TAG_VARIANTS]
    tmpl_vars = [t for t in tmpl_vars if t]
    def has_tag(txt: str) -> bool:
        ids = tok.encode(txt, add_special_tokens=True, truncation=True, max_length=args.max_seq_len)
        pos, _ = find_any_subseq(ids, tmpl_vars)
        return pos != -1

    raw_hit = sum(ANSWER_TAG in r["text"] for r in train_ds) / max(1, len(train_ds))
    trunc_hit = sum(has_tag(r["text"]) for r in train_ds) / max(1, len(train_ds))
    print(f"RAW_TAG_HIT={raw_hit:.3f}  TRUNC_TAG_HIT={trunc_hit:.3f}")
    if trunc_hit == 0.0:
        raise RuntimeError("Không mẫu nào còn TAG sau truncate. Tăng --max-seq-len hoặc kiểm tra TAG/format dữ liệu.")

    collate_fn = make_collate_fn(tok, args.max_seq_len)

    targs = TrainingArguments(
        output_dir=args.out_adapter,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        evaluation_strategy=args.eval_strategy,
        eval_steps=args.save_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        fp16=True, bf16=False,                # T4 dùng fp16
        optim="paged_adamw_8bit",             # optimizer tiết kiệm VRAM (bnb)
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        report_to="none",
        dataloader_num_workers=2,
    )

    print("Start training…")
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds if args.eval_strategy != "no" else None,
        data_collator=collate_fn,
        tokenizer=tok,
    )
    trainer.train()

    print("Saving adapter…")
    model.save_pretrained(args.out_adapter)
    tok.save_pretrained(args.out_adapter)

    if args.no_merge:
        print("Skip merging LoRA (no-merge). Done.")
        return

    # Merge LoRA -> full (rất nặng; có thể fail do RAM). Khuyên dùng adapter cho POC.
    print("Merging LoRA -> full model…")
    try:
        base_fp = AutoModelForCausalLM.from_pretrained(
            args.base,
            cache_dir=args.cache_dir,
            torch_dtype=torch.float16,  # fp16 cho nhẹ
            device_map="cpu",           # merge trên CPU để tránh tràn VRAM
        )
        merged = PeftModel.from_pretrained(base_fp, args.out_adapter).merge_and_unload()
        merged.save_pretrained(args.out_merged)
        tok.save_pretrained(args.out_merged)
        print("Saved merged solver to:", args.out_merged)
    except Exception as e:
        print(
            "\n[WARN] Không thể merge LoRA -> full model (thiếu RAM?). "
            "Bạn vẫn có thể dùng adapter trực tiếp qua PEFT.\n"
            f"Chi tiết: {repr(e)}\n"
        )


if __name__ == "__main__":
    main()
