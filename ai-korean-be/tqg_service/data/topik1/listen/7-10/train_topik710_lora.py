#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TOPIK I (7–10) FROM-SCRATCH GENERATOR — LoRA/QLoRA + marker collator (DDP-safe)
- Sinh JSON duy nhất chứa: dialogue(>=2 dòng), options A–D, answer, question_no (7|8|9|10)...
- Truncation bên trái để luôn giữ marker + JSON đuôi.
- GPU: tận dụng đa GPU với torchrun (DDP); nếu không có GPU sẽ fallback CPU.
- Tối giản cảnh báo deprecated (bỏ dataset_num_proc khỏi SFTConfig; dùng processing_class thay tokenizer).
- DDP-safe: KHÔNG đặt device_map, KHÔNG set_device thủ công; để Accelerate/Trainer tự quản lý.
"""

import os, json, glob, argparse
from typing import Dict, Any, List
from dataclasses import dataclass

# Giảm phân mảnh GPU, tối ưu CPU threads
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count() or 4))

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

# ---------------------------- PROMPT ----------------------------
INSTR = (
    "### Instruction:\n"
    "TOPIK I (초급) 7~10 유형의 ‘여기는 어디입니까?’ 문제를 '처음부터' 생성하세요. 출력은 JSON 한 개만.\n"
    "키/제약:\n"
    "- options: A,B,C,D (서로 달라야 함),\n"
    "- answer: A|B|C|D (options 키 중 하나),\n"
    "- dialogue: [ { speaker: 남자|여자, text: ... }, ... ] 2~6줄,\n"
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
    # Bắt buộc các khóa quan trọng
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


def formatting_func(examples):
    texts = []
    n = len(examples["options"])  # sẽ raise nếu thiếu
    for i in range(n):
        row = {k: examples[k][i] for k in examples}
        texts.append(build_prompt_from_row(row) + build_target_json(row))
    return texts


def load_many_jsonl(paths: List[str]) -> Dataset:
    dsets = []
    for p in paths:
        d = load_dataset("json", data_files=p, split="train")
        dsets.append(d)
    return concatenate_datasets(dsets) if len(dsets) > 1 else dsets[0]


# ---------- Collator: mask loss sau marker ----------
@dataclass
class CompletionCollatorByMarker:
    processing_class: AutoTokenizer  # dùng cùng kiểu tên với TRL mới
    marker_str: str
    marker_variants: List[List[int]]

    def _find_subseq(self, seq, sub):
        L, l = len(seq), len(sub)
        if l == 0 or L < l:
            return -1
        for i in range(L - l, -1, -1):  # từ phải sang trái
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
        input_ids_list = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attn_masks_list = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]

        pad_id = self.processing_class.pad_token_id
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
        attention_mask = pad_sequence(attn_masks_list, batch_first=True, padding_value=0)

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
    ap.add_argument("--output_dir", type=str, default="topik-710-gen-from-scratch")
    ap.add_argument("--max_seq_len", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=128)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--save_steps", type=int, default=300)
    ap.add_argument("--eval_steps", type=int, default=900)
    ap.add_argument("--logging_steps", type=int, default=100)
    ap.add_argument("--save_total_limit", type=int, default=3)
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--resume_from_checkpoint", action="store_true")
    ap.add_argument("--resume_ckpt_path", type=str, default=None)
    # Dataloader / pin memory
    ap.add_argument("--dataloader_num_workers", type=int, default=1)
    ap.add_argument("--pin_memory", action="store_true")
    # LoRA scope
    ap.add_argument("--mlp_lora", action="store_true", help="Thêm gate/up/down_proj vào LoRA (mặc định attention-only)")

    args = ap.parse_args()

    # Phát hiện thiết bị (DDP an toàn: KHÔNG set device_map, KHÔNG set_device)
    has_cuda = torch.cuda.is_available()
    is_cpu = not has_cuda

    if is_cpu:
        args.use_4bit = False
        print("[INFO] Running on CPU. 4-bit disabled.")
    else:
        # Không in local_rank/world_size ở đây; Accelerate sẽ quản lý.
        print("[INFO] Running on GPU(s). 4-bit =", args.use_4bit)

    # Train files
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
    tokenizer.truncation_side = "left"  # giữ marker + JSON đuôi

    # Quant config (GPU+4bit)
    quant_cfg = None
    if args.use_4bit and not is_cpu:
        bf16_ok = torch.cuda.is_bf16_supported()
        compute_dtype = torch.bfloat16 if bf16_ok else torch.float16
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model_load_kwargs = {}
    if quant_cfg is not None:
        model_load_kwargs["quantization_config"] = quant_cfg

    # DDP-safe: không device_map, để Trainer tự move
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=(torch.float32 if is_cpu else None),
        **model_load_kwargs
    )

    # QLoRA prepare (GPU+4bit)
    if args.use_4bit and not is_cpu:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    try:
        model.config.use_cache = False
    except Exception:
        pass

    # LoRA: mặc định attention-only; nếu --mlp_lora thì thêm MLP proj
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

    # Collator & marker
    marker_str = RESPONSE_TAG + "\n{"
    variants_text = [
        RESPONSE_TAG + "\n{",
        "\n" + RESPONSE_TAG + "\n{",
        RESPONSE_TAG + "{",
        " " + RESPONSE_TAG + "\n{",
        "\n" + RESPONSE_TAG + "{",
    ]
    marker_variants = tokenizer(variants_text, add_special_tokens=False)["input_ids"]
    collator = CompletionCollatorByMarker(
        processing_class=tokenizer,
        marker_str=marker_str,
        marker_variants=marker_variants,
    )

    # Precision flags
    bf16_flag = (not is_cpu) and torch.cuda.is_bf16_supported()
    fp16_flag = (not is_cpu) and (not bf16_flag)

    # SFTConfig (để formatting_func hoạt động ổn định → remove_unused_columns=False)
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_seq_length=args.max_seq_len,
        lr_scheduler_type="cosine",
        bf16=bf16_flag,
        fp16=fp16_flag,
        gradient_checkpointing=(not is_cpu),
        gradient_checkpointing_kwargs={"use_reentrant": False} if not is_cpu else None,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_strategy="steps",
        report_to=[],
        packing=False,
        # Eval/best model
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Dataloader / pin memory
        dataloader_num_workers=max(1, args.dataloader_num_workers),
        dataloader_pin_memory=bool(args.pin_memory and not is_cpu),
        # DDP tối ưu
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )

    # EarlyStopping (nếu sẵn có)
    callbacks = []
    if HAVE_EARLY_STOP:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,     # <- thay cho tokenizer=...
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        formatting_func=formatting_func,
        data_collator=collator,
        callbacks=callbacks or None,
    )

    # Sanity check
    try:
        row0 = {k: train_ds[k][0] for k in train_ds.column_names}
        txt0 = build_prompt_from_row(row0) + build_target_json(row0)
        print("[DEBUG] tail preview:", repr(txt0[-150:]))
    except Exception as e:
        print("[WARN] preview failed:", repr(e))

    # Resume
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
