# ============================================
# Merge LoRA adapters (GEN + MCQ) on Colab T4
#  - Tải base về local
#  - Tạo device_map với max_memory
#  - dispatch + offload tránh OOM
# ============================================

# !pip install -q "transformers==4.43.3" "accelerate==0.33.0" "peft==0.12.0" safetensors psutil huggingface_hub

import os, gc, psutil, torch, json
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
from peft import PeftModel

# --------- CONFIG ---------
BASE = "meta-llama/Meta-Llama-3-8B-Instruct"

# GEN
ADAPTER_GEN = "/content/drive/MyDrive/ckpts/topik_gen/llama3-8b/adapter"
OUT_GEN     = "/content/drive/MyDrive/ckpts/topik_gen/llama3-8b/best"

# MCQ
ADAPTER_MCQ = "/content/drive/MyDrive/ckpts/topik_mcq/llama3-8b/adapter"
OUT_MCQ     = "/content/drive/MyDrive/ckpts/topik_mcq/llama3-8b/best"

OFFLOAD = "/content/offload"
os.makedirs(OFFLOAD, exist_ok=True)

# --------- MEMORY BUDGET ---------
def fmt_gib(b): return f"{int(b/(1024**3))}GiB"

if torch.cuda.is_available():
    total_vram = torch.cuda.get_device_properties(0).total_memory
    gpu_budget_bytes = int(total_vram * 0.75)   # chừa 25% cho runtime khác
else:
    gpu_budget_bytes = 0

vm = psutil.virtual_memory()
cpu_budget_bytes = min(int(vm.total * 0.50), 10 * 1024**3)  # cap 10GiB cho Colab

max_memory = {"cpu": fmt_gib(cpu_budget_bytes)}
if torch.cuda.is_available():
    max_memory[0] = fmt_gib(gpu_budget_bytes)   # KHÓA là số 0 (GPU index), không phải "cuda:0"

DTYPE = torch.float16  # T4 không có bf16

print("[INFO] max_memory:", max_memory)
print("[INFO] dtype:", DTYPE, "| offload_folder:", OFFLOAD)

# --------- TẢI BASE VỀ LOCAL & TÌM INDEX ---------
def get_local_checkpoint_dir(repo_id: str) -> tuple[str, str]:
    local_dir = snapshot_download(repo_id, local_files_only=False)  # tải full về /root/.cache...
    # tìm index file
    for fn in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        p = os.path.join(local_dir, fn)
        if os.path.exists(p):
            return local_dir, p
    # nếu mô hình không sharded (1 file)
    for fn in os.listdir(local_dir):
        if fn.endswith(".safetensors") or fn.endswith(".bin"):
            return local_dir, os.path.join(local_dir, fn)
    raise FileNotFoundError("Không tìm thấy index hay weight file trong snapshot của base model.")

# --------- LOAD BASE VỚI DISPATCH + OFFLOAD ---------
def load_base_with_offload():
    local_dir, index_path = get_local_checkpoint_dir(BASE)
    print(f"[INFO] local base dir: {local_dir}")
    print(f"[INFO] index/ckpt    : {index_path}")

    cfg = AutoConfig.from_pretrained(local_dir, trust_remote_code=False)
    with init_empty_weights():
        empty_model = AutoModelForCausalLM.from_config(cfg, torch_dtype=DTYPE, trust_remote_code=False)

    device_map = infer_auto_device_map(
        empty_model,
        dtype=DTYPE,
        no_split_module_classes=["LlamaDecoderLayer"],
        max_memory=max_memory,
    )

    model = load_checkpoint_and_dispatch(
        empty_model,
        checkpoint=index_path,           # <-- đường dẫn LOCAL tới index/ckpt
        device_map=device_map,
        dtype=DTYPE,
        offload_folder=OFFLOAD,
        offload_state_dict=True,
        offload_buffers=True,
    )
    return model

# --------- MERGE 1 ADAPTER ---------
def merge_one(adapter_dir: str, out_dir: str):
    print(f"\n[MERGE] Adapter: {adapter_dir}")
    os.makedirs(out_dir, exist_ok=True)

    base = load_base_with_offload()
    model = PeftModel.from_pretrained(base, adapter_dir)
    merged = model.merge_and_unload()

    tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    merged.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    print(f"[DONE] Saved merged model to: {out_dir}")

    del merged, model, base, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --------- CHẠY: MERGE GEN rồi MCQ ---------
merge_one(ADAPTER_GEN, OUT_GEN)
merge_one(ADAPTER_MCQ, OUT_MCQ)

print("\nAll merges completed ✅")
