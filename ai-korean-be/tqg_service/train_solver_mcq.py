#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train solver MCQ (A/B/C/D) cho TOPIK I 1–4 (không dùng TRL):
- Đầu vào: train/val JSONL từ make_splits_by_pack.py
- Prompt kết thúc bằng @@ANSWER@@ ; chỉ tính loss sau TAG
- LoRA + merge -> full model (model.safetensors)
"""

import os, io, json, argparse, random
from typing import Iterable, Dict, Any

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, PeftModel


ANSWER_TAG = "@@ANSWER@@"

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

def read_jsonl(path: str):
    with io.open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def build_prompt(ex: Dict[str, Any]) -> str:
    inst = ex["instruction_ko"]
    ctx = ex["context_ko"]
    choices = "\n".join(ex["choices"])
    # KẾT THÚC = đúng TAG, không thêm space/newline thừa
    return (
        f"{inst}\n\n"
        f"지문:\n{ctx}\n\n"
        f"보기:\n{choices}\n\n"
        "정답은 A/B/C/D 중 하나의 글자만 쓰세요.\n"
        f"{ANSWER_TAG}"
    )

class ListJsonlDataset(TorchDataset):
    def __init__(self, path: str):
        self.rows = []
        for ex in read_jsonl(path):
            prompt = build_prompt(ex)
            # nối nhãn với 1 space (để model sinh “ C” chẳng hạn)
            self.rows.append({"text": prompt + " " + ex["label"]})
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

def make_collate_fn(tokenizer, max_len: int):
    def _get_text(item):
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            # ưu tiên 'text', fallback vài tên thường gặp
            for k in ("text", "input", "prompt", "source"):
                if k in item:
                    return item[k]
        return str(item)

    def collate(batch):
        texts = [ _get_text(b) for b in batch ]
        enc = tokenizer(
            texts, padding=True, truncation=True, max_length=max_len,
            return_offsets_mapping=True, add_special_tokens=True
        )
        labels = []
        for text, offsets, input_ids in zip(texts, enc["offset_mapping"], enc["input_ids"]):
            tag_pos = text.rfind(ANSWER_TAG)
            if tag_pos == -1:
                labels.append(input_ids)  # hiếm khi không có TAG: học cả chuỗi
                continue
            tag_end = tag_pos + len(ANSWER_TAG)
            lab = []
            for (s, e), tid in zip(offsets, input_ids):
                if s is None or e is None:
                    lab.append(-100)          # special tokens
                else:
                    lab.append(tid if s >= tag_end else -100)
            labels.append(lab)
        enc.pop("offset_mapping")
        to_tensor = lambda x: torch.tensor(x, dtype=torch.long)
        enc = {k: to_tensor(v) for k, v in enc.items()}
        enc["labels"] = to_tensor(labels)
        return enc
    return collate

def main():
    ap = argparse.ArgumentParser("Train solver MCQ (no-TRL)")
    ap.add_argument("--train", default=r"data\topik1\1-4\splits\train.jsonl")
    ap.add_argument("--val",   default=r"data\topik1\1-4\splits\val.jsonl")
    ap.add_argument("--out-adapter", default=r"ckpts\topik_mcq\adapter")
    ap.add_argument("--out-merged",  default=r"ckpts\topik_mcq\best")
    ap.add_argument("--base", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-seq-len", type=int, default=320)
    ap.add_argument("--seed", type=int, default=13)

    # ▼ Thêm các tham số log/save ngắn để thấy kết quả ngay
    ap.add_argument("--max-steps", type=int, default=0)         # 0 = không ép số bước
    ap.add_argument("--logging-steps", type=int, default=1)
    ap.add_argument("--save-steps", type=int, default=10)
    ap.add_argument("--eval-strategy", default="no")            # "no" | "steps"
    args = ap.parse_args()

    os.makedirs(args.out_adapter, exist_ok=True)
    os.makedirs(args.out_merged, exist_ok=True)
    set_seed_all(args.seed)

    print("Loading datasets…")
    train_ds = ListJsonlDataset(args.train)
    val_ds   = ListJsonlDataset(args.val)

    print("Loading tokenizer/model…")
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map="auto",
    )

    # LoRA cho LLaMA-like (TinyLlama)
    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    model = get_peft_model(model, peft_cfg)

    collate_fn = make_collate_fn(tok, args.max_seq_len)

    args_tr = TrainingArguments(
        output_dir=args.out_adapter,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,            # ▼ log dày
        evaluation_strategy=args.eval_strategy,      # ▼ tắt eval cho nhanh (hoặc "steps")
        eval_steps=args.save_steps,                  # đồng bộ mốc với save (nếu bật eval)
        save_steps=args.save_steps,                  # ▼ lưu sớm để thấy file ngay
        bf16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,                 # ▼ giữ cột 'text' cho collate
        dataloader_pin_memory=False,                 # ▼ tắt pin_memory (hết warning)
        max_steps=args.max_steps if args.max_steps > 0 else -1,  # ▼ smoke test ngắn
    )

    print("Start training…")
    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        tokenizer=tok,
    )
    trainer.train()

    print("Saving adapter…")
    model.save_pretrained(args.out_adapter)
    tok.save_pretrained(args.out_adapter)

    print("Merging LoRA -> full model…")
    base_fp = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map="auto",
    )
    merged = PeftModel.from_pretrained(base_fp, args.out_adapter).merge_and_unload()
    merged.save_pretrained(args.out_merged)
    tok.save_pretrained(args.out_merged)
    print(f"Saved merged solver to: {args.out_merged}")

if __name__ == "__main__":
    main()
