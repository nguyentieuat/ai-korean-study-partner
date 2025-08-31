#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fine-tune (LoRA/QLoRA) trên SFT JSONL {"prompt","response"}.
- Chat template của base model (user -> assistant)
- MASK LOSS: chỉ tính loss phần assistant
- Chạy cả local (CPU/không CUDA) và server (CUDA + bitsandbytes)
- Tự tương thích nhiều phiên bản transformers bằng cách lọc kwargs TrainingArguments

Ví dụ:
# Local (không QLoRA), model nhẹ:
python train_lora_sft.py \
  --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --train data/topik1/1-4/dataset_mcq_views/train.rag.sft.jsonl \
  --val   data/topik1/1-4/dataset_mcq_views/val.rag.sft.jsonl \
  --output runs/tinyllama_topik_lora \
  --epochs 3 --per_device_batch_size 4 --grad_accum 16 \
  --max_seq_len 1024 --lr 1e-4 --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --gradient_checkpointing

# Server CUDA + QLoRA 4-bit:
python train_lora_sft.py \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --train data/topik1/1-4/dataset_mcq_views/train.rag.sft.jsonl \
  --val   data/topik1/1-4/dataset_mcq_views/val.rag.sft.jsonl \
  --output runs/llama3_topik_lora \
  --epochs 3 --per_device_batch_size 8 --grad_accum 8 \
  --max_seq_len 1024 --lr 1e-4 --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --bnb_4bit --gradient_checkpointing
"""

import os
import re
import json
import math
import argparse
import random
import inspect
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

# Optional imports guarded at runtime
try:
    from transformers import BitsAndBytesConfig  # type: ignore
except Exception:
    BitsAndBytesConfig = None  # noqa: N816

try:
    from peft import LoraConfig, get_peft_model
except Exception as e:
    raise RuntimeError(
        "PEFT chưa được cài. Hãy chạy: pip install -U peft"
    ) from e


# ---------------------------
# Utils
# ---------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                data.append(json.loads(s))
    return data


# ---------------------------
# Dataset (mask loss outside assistant)
# ---------------------------

class SFTJsonlDataset(Dataset):
    """
    Mỗi dòng: {"prompt": "...", "response": "..."}
    Build hội thoại:
      user: prompt
      assistant: response
    Dùng tokenizer.apply_chat_template để tạo text đầy đủ.
    Mask: label = -100 cho toàn bộ token TRƯỚC khi assistant bắt đầu.
    """

    def __init__(self, data: List[Dict[str, str]], tokenizer, max_len: int):
        self.data = data
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def _build_texts(self, prompt: str, response: str) -> Dict[str, str]:
        conv = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        full_text = self.tok.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=False
        )

        prefix_conv = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": ""},
        ]
        prefix_text = self.tok.apply_chat_template(
            prefix_conv, tokenize=False, add_generation_prompt=False
        )
        return {"full_text": full_text, "prefix_text": prefix_text}

    def __getitem__(self, idx):
        ex = self.data[idx]
        prompt = ex["prompt"]
        response = ex["response"]

        texts = self._build_texts(prompt, response)
        enc_full = self.tok(
            texts["full_text"],
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_len,
        )
        enc_pref = self.tok(
            texts["prefix_text"],
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_len,
        )

        input_ids = enc_full.input_ids
        labels = input_ids.copy()
        start = min(len(enc_pref.input_ids), len(input_ids))
        for i in range(start):
            labels[i] = -100

        pad_id = self.tok.pad_token_id or self.tok.eos_token_id
        attn = [1] * len(input_ids)

        if len(input_ids) < self.max_len:
            pad_len = self.max_len - len(input_ids)
            input_ids = input_ids + [pad_id] * pad_len
            labels = labels + [-100] * pad_len
            attn = attn + [0] * pad_len
        else:
            input_ids = input_ids[: self.max_len]
            labels = labels[: self.max_len]
            attn = attn[: self.max_len]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


# ---------------------------
# Helpers for env-robust TrainingArguments & quantization
# ---------------------------

def filter_kwargs_for_callable(kwargs: Dict[str, Any], fn) -> Dict[str, Any]:
    """Giữ lại các kwargs mà hàm/ctor `fn` chấp nhận (tránh lỗi tham số lạ giữa các version)."""
    try:
        params = set(inspect.signature(fn).parameters.keys())
        return {k: v for k, v in kwargs.items() if k in params and v is not None}
    except Exception:
        # Nếu không introspect được, trả về kwargs không None (best-effort)
        return {k: v for k, v in kwargs.items() if v is not None}


def choose_torch_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        # bf16 nếu GPU compute capability >= 8, else fp16
        try:
            major = torch.cuda.get_device_capability(0)[0]
            if major >= 8:
                return torch.bfloat16
            else:
                return torch.float16
        except Exception:
            return torch.float16
    return torch.float32


def build_quant_config(use_4bit: bool, use_8bit: bool):
    """Trả về (quant_config, use_bnb_flag). Nếu không có CUDA/bnb, fallback None."""
    use_bnb = False
    quant_cfg = None
    if use_4bit or use_8bit:
        if BitsAndBytesConfig is None:
            print("[WARN] transformers không có BitsAndBytesConfig; bỏ quant hóa.")
            return None, False
        try:
            import bitsandbytes as bnb  # noqa: F401
        except Exception as e:
            print(f"[WARN] bitsandbytes không khả dụng ({e}); bỏ quant hóa.")
            return None, False
        if not torch.cuda.is_available():
            print("[WARN] Không có CUDA; bỏ quant hóa 4/8-bit.")
            return None, False
        # OK: có CUDA + bnb
        use_bnb = True
        if use_4bit:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif use_8bit:
            quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
    return quant_cfg, use_bnb


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True, help="VD: meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--train", required=True, help="train jsonl (prompt/response)")
    ap.add_argument("--val", required=True, help="val jsonl (prompt/response)")
    ap.add_argument("--output", required=True, help="output dir")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--per_device_batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--device", type=str, default="auto", help='auto|cpu|cuda|cuda:0 ... (mặc định "auto")')

    # LoRA
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", type=str,
                    default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                    help="comma-separated list")

    # Quantization flags
    ap.add_argument("--bnb_4bit", action="store_true", help="QLoRA 4-bit (yêu cầu CUDA + bitsandbytes)")
    ap.add_argument("--load_in_8bit", action="store_true", help="8-bit (yêu cầu CUDA + bitsandbytes)")

    # Logging / eval
    ap.add_argument("--save_steps", type=int, default=0)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--eval_steps", type=int, default=0, help=">0 để eval theo steps (nếu version hỗ trợ)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)

    # ---------------- Tokenizer ----------------
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # ---------------- Quantization / device ----------------
    torch_dtype = choose_torch_dtype()
    quant_cfg, use_bnb = build_quant_config(args.bnb_4bit, args.load_in_8bit)

    device_map = "auto"
    if args.device.lower() in {"cpu", "cuda"} or args.device.lower().startswith("cuda"):
        device_map = args.device

    # ---------------- Model ----------------
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        device_map=device_map,
        quantization_config=quant_cfg,
    )
    model.config.pad_token_id = tok.pad_token_id
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # ---------------- PEFT LoRA ----------------
    target_modules = [s.strip() for s in args.target_modules.split(",") if s.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)

    # ---------------- Data ----------------
    train_data = read_jsonl(args.train)
    val_data = read_jsonl(args.val)
    train_ds = SFTJsonlDataset(train_data, tok, args.max_seq_len)
    val_ds = SFTJsonlDataset(val_data, tok, args.max_seq_len)

    # ---------------- TrainingArguments (robust across versions) ----------------
    # Chọn optimizer: nếu thật sự dùng bnb (CUDA + bnb) thì paged_adamw_32bit, ngược lại adamw_torch
    optim_name = "paged_adamw_32bit" if use_bnb else "adamw_torch"

    base_kwargs = dict(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=max(1, args.per_device_batch_size // 2),
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=(torch_dtype == torch.float16),
        bf16=(torch_dtype == torch.bfloat16),
        logging_steps=args.logging_steps,
        save_steps=(args.save_steps if args.save_steps > 0 else None),
        save_total_limit=2,
        lr_scheduler_type="cosine",
        report_to="none",
        optim=optim_name,
    )

    # Thêm eval khi được yêu cầu; tên key tùy version (evaluation_strategy hoặc eval_strategy)
    eval_kwargs = {}
    if args.eval_steps and args.eval_steps > 0:
        eval_kwargs["eval_steps"] = args.eval_steps
        # thử cả 2 khóa, sau đó lọc theo chữ ký ctor
        eval_kwargs["evaluation_strategy"] = "steps"
        eval_kwargs["eval_strategy"] = "steps"

    ta_kwargs = {**base_kwargs, **eval_kwargs}
    ta_kwargs = filter_kwargs_for_callable(ta_kwargs, TrainingArguments.__init__)

    train_args = TrainingArguments(**ta_kwargs)

    # ---------------- Trainer ----------------
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds if ("evaluation_strategy" in ta_kwargs or "eval_steps" in ta_kwargs or "eval_strategy" in ta_kwargs) else None,
        tokenizer=tok,
    )

    trainer.train()
    trainer.save_model(args.output)
    tok.save_pretrained(args.output)
    print(f"[OK] Finished training. Saved LoRA to: {args.output}")


if __name__ == "__main__":
    main()
