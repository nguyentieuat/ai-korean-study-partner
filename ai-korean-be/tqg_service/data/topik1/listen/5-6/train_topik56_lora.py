# %%writefile train_topik56_lora.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TOPIK I (5–6) FROM-SCRATCH GENERATOR — LoRA/QLoRA + Robust Custom Collator
- CPU nếu không có GPU; có GPU thì dùng tối đa GPU hiện có (DDP qua torchrun/accelerate).
- Multi-GPU: đặt device theo LOCAL_RANK và load model với device_map={"": local_rank} (tránh OOM dồn vào GPU:0).
- QLoRA 4-bit: thêm bnb_4bit_compute_dtype=float16 (hợp T4).
- TRL cũ: fallback bỏ evaluation_strategy và dùng PeriodicEvalCallback để eval theo --eval_steps.
"""

import os, json, glob, argparse, gc
from typing import Dict, Any, List
from dataclasses import dataclass

# ===== CPU/GPU env =====
CPU_COUNT = os.cpu_count() or 4
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_COUNT))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_COUNT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import torch
torch.set_num_threads(CPU_COUNT)
try:
    torch.set_num_interop_threads(min(16, max(2, CPU_COUNT // 2)))
except Exception:
    pass
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# Early stopping (optional)
try:
    from transformers import EarlyStoppingCallback, TrainerCallback
    HAVE_EARLY_STOP = True
except Exception:
    EarlyStoppingCallback = None
    TrainerCallback = object
    HAVE_EARLY_STOP = False

# ---------------- PROMPT ----------------
INSTR = (
    "### Instruction:\n"
    "TOPIK I (초급) 5~6 유형 문제를 '처음부터' 생성하세요. 출력은 JSON 한 개만.\n"
    "키/제약:\n"
    "- options: A,B,C,D (서로 달라야 함),\n"
    "- answer: A|B|C|D (options 키 중 하나),\n"
    "- dialogue: [ { speaker: 남자|여자, text: ... } ] 단 한 줄,\n"
    "- question_no: 5|6,\n"
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
    qno   = int(row.get("question_no", 5))
    return PROMPT_TMPL.format(instr=INSTR, topic=topic, qno=qno)

def build_target_json(row: Dict[str, Any]) -> str:
    item = {
        "type": "Nghe_Loi_Tiep",
        "section": "Nghe",
        "level": "1",
        "title": "[5~6] 다음을 듣고 <보기>와 같이 이어지는 말을 고르십시오.",
        "question": "이어지는 말을 고르십시오.",
        "explanation": "",
        "score": int(row.get("score", 4)),
        "options": row["options"],
        "answer": row["answer"],
        "dialogue": row["dialogue"],
        "question_no": int(row.get("question_no", 5)),
    }
    return json.dumps(item, ensure_ascii=False)

def formatting_func(examples):
    n = len(examples["options"])
    return [
        build_prompt_from_row({k: examples[k][i] for k in examples}) +
        build_target_json({k: examples[k][i] for k in examples})
        for i in range(n)
    ]

def load_many_jsonl(paths: List[str]) -> Dataset:
    dsets = []
    for p in paths:
        d = load_dataset("json", data_files=p, split="train")
        dsets.append(d)
    return concatenate_datasets(dsets) if len(dsets) > 1 else dsets[0]

# ------------- Collator (mask trước marker) -------------
@dataclass
class CompletionCollatorByMarker:
    tokenizer: AutoTokenizer
    marker_str: str
    marker_variants: List[List[int]]

    def _find_subseq(self, seq, sub):
        L, l = len(seq), len(sub)
        if l == 0 or L < l: return -1
        for i in range(L - l, -1, -1):
            if seq[i:i+l] == sub: return i
        return -1

    def _find_marker_tokenwise(self, ids):
        for sub in self.marker_variants:
            pos = self._find_subseq(ids, sub)
            if pos != -1: return pos + len(sub)
        return -1

    def _find_marker_by_text(self, ids):
        text = self.tokenizer.decode(ids, skip_special_tokens=False)
        idx = text.rfind(self.marker_str)
        if idx == -1: return -1
        prefix = text[:idx + len(self.marker_str)]
        toks = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
        return min(len(toks), len(ids))

    def __call__(self, features):
        input_ids_list = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attn_masks_list = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        pad_id = self.tokenizer.pad_token_id
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
        attention_mask = pad_sequence(attn_masks_list, batch_first=True, padding_value=0)
        labels = input_ids.clone()
        skipped = 0
        for i, f in enumerate(features):
            ids = f["input_ids"]
            start = self._find_marker_tokenwise(ids)
            if start == -1: start = self._find_marker_by_text(ids)
            if start == -1:
                labels[i, :] = -100; skipped += 1
            else:
                labels[i, :start] = -100
        if skipped:
            print(f"[Collator] skipped {skipped}/{len(features)} samples (no marker found)")
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# --------- Fallback eval callback for old TRL ---------
class PeriodicEvalCallback(TrainerCallback):
    def __init__(self, every_steps: int):
        self.every = max(0, int(every_steps))

    def on_step_end(self, args, state, control, **kwargs):
        if self.every and state.global_step and (state.global_step % self.every == 0):
            control.should_evaluate = True
        return control

# ------------------------------- MAIN -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--train_glob", type=str, required=True)
    ap.add_argument("--eval_file", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default="topik-56-gen")
    ap.add_argument("--max_seq_len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=128)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--eval_steps", type=int, default=600)  # dùng cho PeriodicEvalCallback khi TRL cũ
    ap.add_argument("--logging_steps", type=int, default=100)
    ap.add_argument("--save_total_limit", type=int, default=3)
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--dataloader_num_workers", type=int, default=max(1, min(8, CPU_COUNT - 1)))
    ap.add_argument("--dataset_num_proc",   type=int, default=max(1, min(8, CPU_COUNT - 1)))
    ap.add_argument("--pin_memory", action="store_true")
    # Resume
    ap.add_argument("--resume_from_checkpoint", action="store_true")
    ap.add_argument("--resume_ckpt_path", type=str, default=None)

    args = ap.parse_args()

    # Detect layout
    n_gpus = torch.cuda.device_count()
    is_cpu  = (n_gpus == 0)
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = (n_gpus > 1) and (local_rank != -1 or world_size > 1)

    if is_cpu:
        args.use_4bit = False
        print("[INFO] Running on CPU. 4-bit disabled.")
    else:
        if local_rank != -1:
            torch.cuda.set_device(local_rank)  # VERY IMPORTANT: gán device theo rank để tránh dồn vào GPU:0
        if is_distributed:
            print(f"[INFO] Multi-GPU detected. n_gpus={n_gpus}, world_size={world_size}. 4-bit={args.use_4bit}")
        else:
            print(f"[INFO] Single GPU detected. 4-bit={args.use_4bit}")

    # Expand files
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

    # Quantization (4-bit) — T4 không có bf16, dùng fp16 compute dtype
    quant_cfg = None
    if args.use_4bit and not is_cpu:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )

    # Free caches trước khi chuẩn bị mô hình
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load model — tránh OOM trên DDP bằng cách map theo local_rank
    if is_cpu:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.float32, device_map={"": "cpu"}
        )
    else:
        if is_distributed or n_gpus > 1:
            device_map = {"": local_rank if local_rank != -1 else 0}
        else:
            device_map = "auto"
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=quant_cfg,
            device_map=device_map
        )

    # QLoRA prepare (chỉ GPU)
    if args.use_4bit and not is_cpu:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    try:
        model.config.use_cache = False
    except Exception:
        pass

    # LoRA
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # Debug trainable params
    try:
        trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in model.parameters())
        print(f"[DEBUG] trainable params: {trn:,} / {tot:,} ({trn/tot:.4%})")
    except Exception:
        pass

    # Collator
    marker_str = RESPONSE_TAG + "\n{"
    variants_text = [
        RESPONSE_TAG + "\n{",
        "\n" + RESPONSE_TAG + "\n{",
        RESPONSE_TAG + "{",
        " " + RESPONSE_TAG + "\n{",
        "\n" + RESPONSE_TAG + "{",
    ]
    marker_variants = [tokenizer.encode(t, add_special_tokens=False) for t in variants_text]
    collator = CompletionCollatorByMarker(tokenizer=tokenizer, marker_str=marker_str, marker_variants=marker_variants)

    # Precision flags
    if is_cpu:
        bf16_flag = False; fp16_flag = False
    else:
        bf16_flag = torch.cuda.is_bf16_supported()
        fp16_flag = not bf16_flag  # T4 → fp16

    # Try full-featured SFTConfig; fallback nếu TRL cũ
        # ---------- SFTConfig + Callbacks (an toàn với TRL cũ) ----------
    callbacks = []

    # Thử cấu hình “đầy đủ” trước; nếu TypeError (TRL cũ) thì fallback an toàn.
    eval_in_config = True
    try:
        sft_config = SFTConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            warmup_ratio=args.warmup_ratio,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            max_seq_length=args.max_seq_len,
            lr_scheduler_type="cosine",
            bf16=(not is_cpu and torch.cuda.is_bf16_supported()),
            fp16=(not is_cpu and not torch.cuda.is_bf16_supported()),
            gradient_checkpointing=not is_cpu,
            gradient_checkpointing_kwargs={"use_reentrant": False} if not is_cpu else None,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            save_strategy="steps",
            # Các khóa dưới đây có thể KHÔNG tồn tại ở TRL cũ:
            evaluation_strategy="steps",
            eval_steps=args.eval_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=[],
            packing=False,
            dataloader_num_workers=max(1, args.dataloader_num_workers),
            dataloader_pin_memory=bool(args.pin_memory and not is_cpu),
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
        )

        # Chỉ add EarlyStopping khi chắc chắn cả hai khóa trên hiện DIỆN và hợp lệ:
        can_early_stop = (
            HAVE_EARLY_STOP
            and getattr(sft_config, "load_best_model_at_end", False) is True
            and getattr(sft_config, "metric_for_best_model", None) is not None
        )
        if can_early_stop:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
        else:
            print("[INFO] EarlyStopping disabled (missing load_best_model_at_end/metric_for_best_model).")

    except TypeError:
        # ===== Fallback an toàn cho TRL cũ: KHÔNG dùng early stopping của HF =====
        eval_in_config = False
        print("[WARN] TRL cũ: bỏ evaluation_strategy/load_best_model_at_end/metric_for_best_model trong SFTConfig.")
        from transformers import TrainerCallback
        class PeriodicEvalCallback(TrainerCallback):
            def __init__(self, every_steps: int): self.every = max(0, int(every_steps))
            def on_step_end(self, args, state, control, **kwargs):
                if self.every and state.global_step and (state.global_step % self.every == 0):
                    control.should_evaluate = True
                return control

        sft_config = SFTConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            warmup_ratio=args.warmup_ratio,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            max_seq_length=args.max_seq_len,
            lr_scheduler_type="cosine",
            bf16=(not is_cpu and torch.cuda.is_bf16_supported()),
            fp16=(not is_cpu and not torch.cuda.is_bf16_supported()),
            gradient_checkpointing=not is_cpu,
            gradient_checkpointing_kwargs={"use_reentrant": False} if not is_cpu else None,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            save_strategy="steps",
            report_to=[],
            packing=False,
            dataloader_num_workers=max(1, args.dataloader_num_workers),
            dataloader_pin_memory=bool(args.pin_memory and not is_cpu),
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
        )
        # TỰ eval theo chu kỳ, KHÔNG add EarlyStopping của HF để tránh assert:
        callbacks = [PeriodicEvalCallback(args.eval_steps)]

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        formatting_func=formatting_func,
        data_collator=collator,
        callbacks=callbacks if callbacks else None,
        dataset_num_proc=max(1, args.dataset_num_proc),
    )

    # Preview 1 mẫu
    try:
        row0 = {k: train_ds[k][0] for k in train_ds.column_names}
        txt0 = build_prompt_from_row(row0) + build_target_json(row0)
        print("[DEBUG] tail preview:", repr(txt0[-150:]))
    except Exception:
        pass

    # Resume
    resume_arg = None
    if args.resume_ckpt_path:
        resume_arg = args.resume_ckpt_path
    elif args.resume_from_checkpoint:
        resume_arg = True

    print("[INFO] Start training...")
    if resume_arg is None:
        trainer.train()
    else:
        print(f"[INFO] Resuming from checkpoint = {resume_arg}")
        trainer.train(resume_from_checkpoint=resume_arg)

    # Nếu TRL cũ, vẫn evaluate cuối
    if not eval_in_config:
        try:
            print("[INFO] Final evaluate():")
            print(trainer.evaluate())
        except Exception as e:
            print("[WARN] evaluate() skipped:", e)

    print("[INFO] Saving adapter to", args.output_dir)
    trainer.save_model()

if __name__ == "__main__":
    import sys, traceback
    try:
        main()
    except SystemExit as e:
        print(f"[FATAL] SystemExit code={getattr(e,'code',None)}", file=sys.stderr)
        traceback.print_exc()
        raise
    except Exception as e:
        print(f"[FATAL] {e!r}", file=sys.stderr)
        traceback.print_exc()
        os._exit(1)
