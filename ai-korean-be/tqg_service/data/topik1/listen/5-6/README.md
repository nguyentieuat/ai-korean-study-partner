# TOPIK 5–6 LoRA (Train riêng)

## Bước 1 — Tạo môi trường
```bash
python -m venv .venv && source .venv/bin/activate  # Windows dùng .venv\Scripts\activate
pip install -r requirements.txt
```
> Windows không GPU: có thể bỏ 4-bit bằng cách sửa script (set `load_in_4bit=False` và gỡ bitsandbytes).

## Bước 2 — Chuẩn bị dữ liệu
Tạo thư mục:
```
data/train/   # đặt tất cả các file *.jsonl dùng để train
data/dev/     # ví dụ: topik_I_5_6.jsonl (26 mẫu gốc)
```
Mỗi dòng JSONL cần có: `options{A..D}`, `answer`, `dialogue[{"speaker","text"}]`, `question_no ∈ {5,6}`.

Kiểm tra nhanh:
```bash
python validate_topik56.py dataset/topik_5_6_part1.jsonl
```

## Bước 3 — Train LoRA (QLoRA)
```bash
!# ví dụ 4 GPU
torchrun --nproc_per_node=4 train_topik56_lora_multigpu.py \
  --base_model "meta-llama/Meta-Llama-3-8B-Instruct" \
  --train_glob "/path/train.jsonl" \
  --eval_file  "/path/dev.jsonl" \
  --output_dir "out/topik56-lora-mgpu" \
  --max_seq_len 256 --epochs 3 --lr 5e-5 --warmup_ratio 0.1 \
  --batch_size 4 --grad_accum 32 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --save_steps 200 --eval_steps 200 --logging_steps 50 \
  --save_total_limit 3 \
  --use_4bit \
  --dataloader_num_workers 4 \
  --dataset_num_proc 4 \
  --pin_memory
  --resume_ckpt_path out/topik710_lora4bit/checkpoint-1200


```

## Bước 4 — Suy luận & Router
```bash
BASE=meta-llama/Meta-Llama-3-8B-Instruct ADAPTER_14=nguyentieuat/topik-mcq-llama3-8b-lora ADAPTER_56=nguyentieuat/topik-56-llama3-8b-lora python infer_router_demo.py
```
