# %%writefile train_topik710_lora.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json, glob, argparse, gc
from typing import Dict, Any, List
from dataclasses import dataclass

# --- Memory & threads ---
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch.nn.utils.rnn import pad_sequence

from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

try:
    from transformers import EarlyStoppingCallback
    HAVE_EARLY_STOP = True
except Exception:
    EarlyStoppingCallback = None
    HAVE_EARLY_STOP = False

# ---------------- PROMPT ----------------
INSTR = (
    "### Instruction:\n"
    "TOPIK I (초급) 7~10 유형의 ‘여기는 어디입니까?’ 문제를 '처음부터' 생성하세요. 출력은 JSON 한 개만.\n"
    "키/제약:\n"
    "- options: A,B,C,D (서로 달라야 함),\n"
    "- answer: A|B|C|D (options 키 중 하나),\n"
    "- dialogue: [ { speaker: 남자|여자, text: ... }, ... ] 2줄,\n"
    "- question_no: 7|8|9|10,\n"
    "- JSON 외의 텍스트 출력 금지.\n"
)
RESPONSE_TAG = "#### JSON_START ####"

PROMPT_TMPL = (
    "{instr}\n\n"
    "### Controls:\n"
    "topic: {topic}\n"
    "question_no: {qno}\n\n"
    f"{RESPONSE_TAG}\n"
)

def build_prompt_from_row(row: Dict[str, Any]) -> str:
    topic = row.get("topic", "misc")
    qno   = int(row.get("question_no", 7))
    return PROMPT_TMPL.format(instr=INSTR, topic=topic, qno=qno)

def build_target_json(row: Dict[str, Any]) -> str:
    for k in ("options", "answer", "dialogue", "question_no"):
        if k not in row:
            raise ValueError(f"Missing required field in data: {k}")
    item = {
        "type": row.get("type", "Nghe_DiaDiem"),
        "section": row.get("section", "Nghe"),
        "level": row.get("level", "1"),
        "title": row.get("title", "[7~10] 여기는 어디입니까? <보기>와 같이 알맞은 것을 고르십시오."),
        "question": row.get("question", "여기는 어디입니까?"),
        "explanation": row.get("explanation", ""),
        "score": int(row.get("score", 3)),
        "options": row["options"],
        "answer": row["answer"],
        "dialogue": row["dialogue"],
        "question_no": int(row.get("question_no", 7)),
    }
    return json.dumps(item, ensure_ascii=False)

def formatting_func(example: Dict[str, Any]) -> str:
    """
    TRL 0.23 gọi với batched=False ⇒ example là một bản ghi đơn (scalar fields).
    Trả về đúng 1 chuỗi: prompt + JSON target.
    """
    # Kiểm tra tối thiểu để thông báo lỗi rõ ràng
    for k in ("options", "answer", "dialogue", "question_no"):
        if k not in example:
            raise ValueError(f"[formatting_func] Missing required field: {k}")

    prompt = build_prompt_from_row(example)
    target = build_target_json(example)  # đã json.dumps và ensure_ascii=False ở trong
    return prompt + target

def load_many_jsonl(paths: List[str]) -> Dataset:
    dsets = []
    for p in paths:
        d = load_dataset("json", data_files=p, split="train")
        dsets.append(d)
    return concatenate_datasets(dsets) if len(dsets) > 1 else dsets[0]

# --------- Collator: mask loss sau marker ---------
@dataclass
class CompletionCollatorByMarker:
    processing_class: AutoTokenizer
    marker_str: str
    marker_variants: List[List[int]]

    def _find_subseq(self, seq, sub):
        L, l = len(seq), len(sub)
        if l == 0 or L < l:
            return -1
        for i in range(L - l, -1, -1):
            if seq[i:i+l] == sub:
                return i
        return -1

    def _find_marker_tokenwise(self, ids):
        for sub in self.marker_variants:
            pos = self._find_subseq(ids, sub)
            if pos != -1:
                return pos + len(sub)
        return -1

    def _find_marker_by_text(self, ids):
        text = self.processing_class.decode(ids, skip_special_tokens=False)
        idx = text.rfind(self.marker_str)
        if idx == -1:
            return -1
        prefix = text[:idx + len(self.marker_str)]
        toks = self.processing_class(prefix, add_special_tokens=False)["input_ids"]
        return min(len(toks), len(ids))

    def __call__(self, features):
        # Lấy pad_id trước
        pad_id = self.processing_class.pad_token_id

        input_ids_list = []
        attn_masks_list = []

        for f in features:
            ids = f["input_ids"]
            # Một số build TRL chỉ đưa input_ids → tự tạo attention_mask = 1
            am  = f.get("attention_mask", [1] * len(ids))

            input_ids_list.append(torch.tensor(ids, dtype=torch.long))
            attn_masks_list.append(torch.tensor(am,  dtype=torch.long))

        # Pad batch
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
        attention_mask = pad_sequence(attn_masks_list, batch_first=True, padding_value=0)

        # Mask loss trước marker
        labels = input_ids.clone()
        skipped = 0
        for i, f in enumerate(features):
            ids = f["input_ids"]
            start = self._find_marker_tokenwise(ids)
            if start == -1:
                start = self._find_marker_by_text(ids)
            if start == -1:
                labels[i, :] = -100
                skipped += 1
            else:
                labels[i, :start] = -100

        if skipped > 0:
            print(f"[Collator] skipped {skipped}/{len(features)} samples (no marker found)")
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--train_glob", type=str, required=True)
    ap.add_argument("--eval_file", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default="topik-710-llama3-8b-lora")
    ap.add_argument("--max_seq_len", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=7e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=32)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--save_steps", type=int, default=50)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--save_total_limit", type=int, default=3)
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--resume_from_checkpoint", action="store_true")
    ap.add_argument("--resume_ckpt_path", type=str, default=None)
    ap.add_argument("--dataloader_num_workers", type=int, default=4)
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--mlp_lora", action="store_true", help="Thêm gate/up/down_proj vào LoRA (mặc định attention-only)")
    ap.add_argument("--shard_model", action="store_true",
                    help="Shard model qua nhiều GPU bằng device_map='auto' (1 tiến trình).")

    args = ap.parse_args()

    # Guard: không cho chạy DDP cùng --shard_model
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.shard_model and world_size > 1:
        raise SystemExit("❌ Do not use torchrun/DDP with --shard_model. Run a single process (python ...) instead.")

    has_cuda = torch.cuda.is_available()
    is_cpu = not has_cuda
    print("[INFO] CUDA available:", has_cuda)
    if is_cpu:
        args.use_4bit = False
        print("[INFO] Running on CPU. 4-bit disabled.")
    else:
        print("[INFO] Running on GPU(s). 4-bit =", args.use_4bit)

    # Datasets
    train_files = [args.train_glob] if os.path.isfile(args.train_glob) else sorted(glob.glob(args.train_glob))
    if not train_files:
        raise FileNotFoundError(f"No files matched: {args.train_glob}")
    print(f"[INFO] Train files: {len(train_files)}")
    train_ds = load_many_jsonl(train_files)
    eval_ds  = load_dataset("json", data_files=args.eval_file, split="train")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    # BitsAndBytes 4-bit (T4 → fp16 compute)
    quant_cfg = None
    if args.use_4bit and not is_cpu:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model_load_kwargs = {}
    if quant_cfg is not None:
        model_load_kwargs["quantization_config"] = quant_cfg

    device_map_kw = {}
    if args.shard_model:
        device_map_kw["device_map"] = "auto"  # shard qua CUDA_VISIBLE_DEVICES

    # Load base
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=(torch.float32 if is_cpu else None),
        **model_load_kwargs,
        **device_map_kw,
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Low-mem prepare (defer checkpointing)
    if args.use_4bit and not is_cpu:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=False
        )

    # LoRA scope
    target_modules = ["q_proj","k_proj","v_proj","o_proj"]
    if args.mlp_lora:
        target_modules += ["gate_proj","up_proj","down_proj"]

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # Enable checkpointing AFTER LoRA attach
    if not is_cpu:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    try:
        model.config.use_cache = False
    except Exception:
        pass

    # Collator
    marker_str = RESPONSE_TAG + "\n{"
    variants_text = [RESPONSE_TAG + "\n{", "\n" + RESPONSE_TAG + "\n{", RESPONSE_TAG + "{", " " + RESPONSE_TAG + "\n{", "\n" + RESPONSE_TAG + "{"]
    marker_variants = tokenizer(variants_text, add_special_tokens=False)["input_ids"]
    collator = CompletionCollatorByMarker(
        processing_class=tokenizer,
        marker_str=marker_str,
        marker_variants=marker_variants,
    )

    # ---------- SFTConfig (Transformers 4.56 / TRL 0.23 compatible) ----------
    common_cfg = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lr_scheduler_type="cosine",
        bf16=(not is_cpu and torch.cuda.is_bf16_supported()),
        fp16=(not is_cpu and not torch.cuda.is_bf16_supported()),
        gradient_checkpointing=(not is_cpu),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_strategy="steps",
        report_to=[],
        packing=False,
        # đổi evaluation_strategy -> eval_strategy
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=max(1, args.dataloader_num_workers),
        dataloader_pin_memory=bool(args.pin_memory and not is_cpu),
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )

    try:
        # Thử bản có hỗ trợ max_seq_length + gradient_checkpointing_kwargs
        sft_config = SFTConfig(
            max_seq_length=args.max_seq_len,
            gradient_checkpointing_kwargs={"use_reentrant": False} if not is_cpu else None,
            **common_cfg,
        )
    except TypeError:
        try:
            # Bản không có max_seq_length / gradient_checkpointing_kwargs
            sft_config = SFTConfig(**common_cfg)
            print("[INFO] SFTConfig fallback: dropped max_seq_length / gradient_checkpointing_kwargs")
        except TypeError as e:
            # Fallback tối giản: bỏ luôn các trường có thể gây lỗi hiếm gặp
            minimal_cfg = dict(
                output_dir=args.output_dir,
                num_train_epochs=args.epochs,
                per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=args.grad_accum,
                learning_rate=args.lr,
                lr_scheduler_type="cosine",
                save_steps=args.save_steps,
                logging_steps=args.logging_steps,
                report_to=[],
            )
            sft_config = SFTConfig(**minimal_cfg)
            print("[INFO] SFTConfig minimal fallback:", repr(e))

    # Trainer (fallback processing_class/tokenizer)
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)] if HAVE_EARLY_STOP else None
    try:
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=sft_config,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            formatting_func=formatting_func,
            data_collator=collator,
            callbacks=callbacks,
        )
    except TypeError:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=sft_config,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            formatting_func=formatting_func,
            data_collator=collator,
            callbacks=callbacks,
        )

    # Sanity
    try:
        row0 = {k: train_ds[k][0] for k in train_ds.column_names}
        txt0 = build_prompt_from_row(row0) + build_target_json(row0)
        print("[DEBUG] tail preview:", repr(txt0[-150:]))
    except Exception as e:
        print("[WARN] preview failed:", repr(e))

    resume_arg = args.resume_ckpt_path if args.resume_ckpt_path else (True if args.resume_from_checkpoint else None)
    print("[INFO] Start training…")
    if resume_arg is None:
        trainer.train()
    else:
        print(f"[INFO] Resuming from checkpoint = {resume_arg}")
        trainer.train(resume_from_checkpoint=resume_arg)

    print("[INFO] Saving adapter to", args.output_dir)
    trainer.save_model()

if __name__ == "__main__":
    main()
