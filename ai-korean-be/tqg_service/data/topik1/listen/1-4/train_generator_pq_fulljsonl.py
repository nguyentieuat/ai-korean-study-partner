# %%writefile /content/train_generator_pq_fulljsonl.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train GENERATOR (pq_generate_full) với QLoRA 4-bit (tiết kiệm VRAM T4 16GB).

- Input: full.jsonl (mỗi dòng có keys: view="pq_generate_full", controls, seed, assistant_output)
- Prompt kết thúc bằng TAG "<OUT>" ; chỉ tính loss sau TAG (mask prefix)
- LoRA nhẹ: r=8, alpha=16, dropout 0.05, chỉ q/k/v/o proj
- Attention SDPA, gradient checkpointing (non-reentrant)
- Optim paged_adamw_8bit, fp16=True (T4 không có bf16)

Khuyến nghị chạy:
python -u train_generator_pq_fulljsonl.py \
  --train "/path/full.jsonl" --val "/path/full.jsonl" \
  --out-adapter "/path/ckpts/gen/adapter" \
  --out-merged "/path/ckpts/gen/best" \
  --base "meta-llama/Meta-Llama-3-8B-Instruct" \
  --epochs 1 --bsz 1 --grad-accum 32 --max-seq-len 384 --seed 13 \
  --logging-steps 10 --eval-strategy no --no-merge
"""

import os, io, json, argparse, random
from typing import Dict, Any, Iterable, List, Tuple

import torch
from torch.utils.data import Dataset as TorchDataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)

# ======== TAG & biến thể để mask ========
OUT_TAG = "<OUT>"
TAG_VARIANTS = [OUT_TAG, "\n" + OUT_TAG, " " + OUT_TAG]

# ======== Utils ========
def set_seed_all(seed:int=13):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with io.open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def build_prompt(ex: Dict[str, Any]) -> str:
    """
    Dùng controls + seed làm input; yêu cầu trả về assistant_output dưới dạng JSON.
    """
    controls = ex.get("controls", {})
    seed     = ex.get("seed", {})
    # Chỉ in gọn gàng & cố định cấu trúc:
    ctrl_s = json.dumps(controls, ensure_ascii=False)
    seed_s = json.dumps(seed, ensure_ascii=False)

    sys_inst = (
        "Bạn là bộ sinh đề TOPIK I (pq_generate_full). "
        "Hãy trả về trường JSON 'assistant_output' gồm: "
        "{passage_ko: string, questions: ["
        "{prompt_ko: string, choices:[{label:'A'|'B'|'C'|'D', text_ko:string},...], "
        "answer: 'A'|'B'|'C'|'D', rationale_ko:string}]}. "
        "Giữ văn phong ngắn gọn, từ vựng mức TOPIK I. Chỉ tiếng Hàn ở các field *_ko."
    )

    user_inst = (
        f"VIEW: pq_generate_full\n"
        f"CONTROLS: {ctrl_s}\n"
        f"SEED: {seed_s}\n"
        f"Trả về đúng cấu trúc JSON cho 'assistant_output'.\n"
        f"{OUT_TAG}"
    )
    return f"{sys_inst}\n\n{user_inst}"

def json_target(ex: Dict[str, Any]) -> str:
    ao = ex.get("assistant_output", {})
    return json.dumps(ao, ensure_ascii=False, separators=(",", ":"))

class FullJsonlDataset(TorchDataset):
    def __init__(self, path: str):
        self.rows: List[Dict[str, str]] = []
        for ex in read_jsonl(path):
            if ex.get("view") != "pq_generate_full":
                # bỏ các view khác nếu có
                continue
            prompt = build_prompt(ex)
            target = json_target(ex)  # ground truth JSON assistant_output
            # thêm 1 space để tách
            self.rows.append({"text": prompt + " " + target})

    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

def find_any_subseq(hay: List[int], needles: List[List[int]]) -> Tuple[int, int]:
    """
    Tìm vị trí sớm nhất của BẤT KỲ biến thể TAG nào trong chuỗi token 'hay'.
    Trả về (pos_bat_dau, do_dai_mau) hoặc (-1, 0) nếu không thấy.
    """
    best = (len(hay) + 1, 0)
    for nd in needles:
        L = len(nd)
        if L == 0:
            continue
        for i in range(0, len(hay) - L + 1):
            if hay[i:i+L] == nd:
                if i < best[0]:
                    best = (i, L)
                break
    if best[0] <= len(hay):
        return best
    return (-1, 0)

def make_collate_fn(tok, max_len:int):
    template_variants = [tok.encode(v, add_special_tokens=False) for v in TAG_VARIANTS]
    template_variants = [ids for ids in template_variants if len(ids) > 0]
    assert len(template_variants) > 0, "Tokenizer không mã hoá được TAG; hãy đổi OUT_TAG/TAG_VARIANTS."

    def _get_text(item):
        if isinstance(item, str): return item
        if isinstance(item, dict):
            for k in ("text","input","prompt","source"):
                if k in item: return item[k]
        return str(item)

    def collate(batch):
        texts = [_get_text(b) for b in batch]
        enc = tok(
            texts,
            padding=True, truncation=True, max_length=max_len,
            add_special_tokens=True, return_tensors=None
        )
        input_ids = enc["input_ids"]
        attn      = enc["attention_mask"]
        labels, hits = [], 0
        for ids in input_ids:
            pos, L = find_any_subseq(ids, template_variants)
            if pos == -1:
                labels.append([-100]*len(ids))
            else:
                start = pos + L
                lab = [-100]*start + ids[start:]
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

# ======== Main ========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val",   required=True)
    ap.add_argument("--out-adapter", required=True)
    ap.add_argument("--out-merged",  required=True)
    ap.add_argument("--base", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bsz", type=int, default=1)       # T4: 1
    ap.add_argument("--grad-accum", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-seq-len", type=int, default=384)  # hạ để đỡ OOM
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--eval-strategy", default="no")    # "no" | "steps" | "epoch"
    ap.add_argument("--cache-dir", default="/content/cache/hf")
    ap.add_argument("--no-merge", action="store_true", help="Chỉ lưu adapter, bỏ bước merge LoRA.")
    args = ap.parse_args()

    os.makedirs(args.out_adapter, exist_ok=True)
    os.makedirs(args.out_merged,  exist_ok=True)
    set_seed_all(args.seed)

    print("Loading datasets…")
    train_ds = FullJsonlDataset(args.train)
    val_ds   = FullJsonlDataset(args.val)
    print(f"Train size = {len(train_ds)} ; Val size = {len(val_ds)}")

    print("Loading tokenizer/model…")
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True, cache_dir=args.cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ======== QLoRA 4-bit ========
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        cache_dir=args.cache_dir,
        device_map="auto",
        quantization_config=bnb,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )

    # Attention ít VRAM (T4 không FlashAttn2)
    try:
        model.config.attn_implementation = "sdpa"
    except Exception:
        pass

    # Chuẩn bị k-bit + bật gradient checkpointing (non-reentrant)
    model = prepare_model_for_kbit_training(model)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "config"):
        try:
            model.config.use_cache = False
        except Exception:
            pass

    # LoRA nhẹ
    peft_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
    )
    model = get_peft_model(model, peft_cfg)

    # Sanity: HIT rate TAG sau khi tokenize (raw và trunc)
    def hit_rate(dataset, max_len):
        template_variants = [tok.encode(v, add_special_tokens=False) for v in TAG_VARIANTS]
        template_variants = [ids for ids in template_variants if len(ids) > 0]
        def _has_tag_ids(ids):
            pos, L = find_any_subseq(ids, template_variants)
            return pos != -1
        # raw
        raw_hits = 0
        trunc_hits = 0
        for r in dataset:
            txt = r["text"]
            ids_raw = tok.encode(txt, add_special_tokens=True)
            if _has_tag_ids(ids_raw): raw_hits += 1
            ids_trunc = tok.encode(txt, add_special_tokens=True, truncation=True, max_length=max_len)
            if _has_tag_ids(ids_trunc): trunc_hits += 1
        n = max(1, len(dataset))
        return raw_hits/n, trunc_hits/n

    raw_hit, trunc_hit = hit_rate(train_ds, args.max_seq_len)
    print(f"HIT_RATE={trunc_hit:.3f} (truncated), RAW={raw_hit:.3f}")

    collate_fn = make_collate_fn(tok, args.max_seq_len)

    targs = TrainingArguments(
        output_dir=args.out_adapter,
        per_device_train_batch_size=args.bsz,    # =1 cho T4 + 8B
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

        # Mixed-precision + optimizer tiết kiệm VRAM
        fp16=True, bf16=False,
        optim="paged_adamw_8bit",

        # Trainer / Dataloader tối giản copy
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
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

    # CẢNH BÁO: Merge 8B về full model cần RAM lớn; Colab T4/ram 12GB có thể thiếu
    print("Merging LoRA -> full model (có thể tốn nhiều RAM CPU)…")
    base_fp = AutoModelForCausalLM.from_pretrained(
        args.base,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    merged = PeftModel.from_pretrained(base_fp, args.out_adapter).merge_and_unload()
    merged.save_pretrained(args.out_merged)
    tok.save_pretrained(args.out_merged)
    print("Saved merged generator to:", args.out_merged)

if __name__ == "__main__":
    main()
