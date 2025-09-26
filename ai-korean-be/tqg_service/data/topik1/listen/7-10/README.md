# TOPIK I 7–10 Generator (LoRA/QLoRA)

This repo trains a generator to create TOPIK I (questions 7–10) multiple‑choice listening items that output **valid JSON** with: `dialogue`, `options`, `answer`, `question_no`, etc.

## 0) Environment
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements_topik710.txt
```
> **Windows CPU**: omit 4-bit by not passing `--use_4bit` when training.  
> **Windows GPU**: `bitsandbytes` may not be supported; consider Linux or use fp16/bf16 instead.

## 1) Data layout
Put your JSONL files (the 20k we generated, or any compatible set) in one folder, e.g.:
```
data/topik1/listen/7-10/
  topik_listening_20k_with_topic_part1.jsonl
  topik_listening_20k_with_topic_part2.jsonl
  topik_listening_20k_with_topic_part3.jsonl
  topik_listening_20k_with_topic_part4.jsonl
```
Each line must match this schema (keys shown):  
`type, section, level, title, question, explanation, score, options{A..D}, answer, dialogue[{speaker,text}], question_no(7|8|9|10), topic`.

## 2) Merge + dedupe + split
```bash
python merge_and_split_topik710.py   --src_dir data/topik1/listen/7-10   --out_dir data/topik1/listen/7-10/splits   --ratios 0.9 0.05 0.05 --seed 42
```
This creates `train.jsonl`, `dev.jsonl`, `test.jsonl` under `splits/` with stratified balancing by `(topic|question_no|answer|first-speaker)`.

## 3) Train LoRA / QLoRA
Example with Llama‑3‑8B‑Instruct and 4‑bit:
```bash
# Gọn cảnh báo mạng (tuỳ)
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=1
export TORCH_CPP_LOG_LEVEL=ERROR
export NCCL_DEBUG=ERROR

export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 topik710_generator_lora_ddp.py \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --train_glob "data/train/*.jsonl" \
  --eval_file "data/eval/dev.json"l \
  --output_dir out/topik710_ddp2_4bit \
  --use_4bit \
  --batch_size 2 --grad_accum 64 \
  --max_seq_len 512 \
  --epochs 2 --lr 2e-4 --warmup_ratio 0.1 \
  --save_steps 300 --eval_steps 900 --logging_steps 50 \
  --mlp_lora \
  --dataloader_num_workers 2 --pin_memory
  --resume_from_checkpoint/--resume_ckpt_path out/topik710_lora4bit/checkpoint-1200

```
Tips:
- Low VRAM → keep `--use_4bit`, raise `--grad_accum`, keep `--batch_size=2`.
- If loss plateaus: increase `--epochs` to 3, lower `--lr` to `1e-4`, or raise `--warmup_ratio` to `0.2`.

## 4) Validate splits
```bash
python validate_topik710.py data/topik1/listen/7-10/splits/train.jsonl
python validate_topik710.py data/topik1/listen/7-10/splits/dev.jsonl
python validate_topik710.py data/topik1/listen/7-10/splits/test.jsonl
```

## 5) Smoke test after training
```bash
python smoke_test_topik710.py   --ckpt out/topik_710_gen   --num 80 --temperature 0.7   --outfile smoke_topik710_outputs.jsonl   --report smoke_topik710_report.txt
```
This will generate N samples across topics, extract JSON, validate schema, and write a quick report.

## Files in this package
- `merge_and_split_topik710.py` — merge, dedupe, stratified split.
- `train_topik710_lora.py` — LoRA/QLoRA SFT trainer.
- `validate_topik710.py` — schema validator.
- `smoke_test_topik710.py` — post‑train smoke test (already provided).
- `requirements_topik710.txt`, `README_topik710.md` — you are here.

---

### Notes
- Set `HF_TOKEN` / login to access gated models if needed.
- If you use a different base model, ensure it supports chat templates or adapt the trainer formatting accordingly.
