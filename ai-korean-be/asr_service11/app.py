# app.py
import os, asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import torch, whisper

from api_asrSTT import router as asr_router

# ==== Detect device ====
USE_GPU = torch.cuda.is_available()
DEVICE  = "cuda" if USE_GPU else "cpu"

# ==== Model & concurrency presets ====
def _default_model_for_device() -> str:
    # Cho GPU >= 12GB thì ưu tiên large-v3; GPU 8-12GB: medium/small; CPU: base/small
    env = os.getenv("WHISPER_MODEL")
    if env:
        return env
    if USE_GPU:
        # Nhẹ nhàng & nhanh cho serve: "small" (an toàn VRAM ~2.5GB)
        return "small"
    else:
        # CPU nên nhỏ để latency hợp lý
        return "base"

WHISPER_MODEL = _default_model_for_device()

def _default_concurrency() -> int:
    env = os.getenv("MAX_CONCURRENCY")
    if env: return int(env)
    if USE_GPU:
        return 1  # 1–2 tuỳ GPU; 1 là an toàn
    else:
        # CPU: tận dụng đa luồng. Lấy n_threads/2 (tối đa 4–6 cho máy phổ thông)
        cores = os.cpu_count() or 4
        return max(1, min(6, cores // 2))

MAX_CONCURRENCY = _default_concurrency()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Threads & math modes
    if USE_GPU:
        # TF32 có thể giúp nhanh hơn trên Ampere+
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    else:
        # Giới hạn số thread để tránh oversubscribe
        n = max(1, min((os.cpu_count() or 4), MAX_CONCURRENCY * 2))
        try:
            torch.set_num_threads(n)
        except Exception:
            pass
        os.environ.setdefault("OMP_NUM_THREADS", str(n))
        os.environ.setdefault("MKL_NUM_THREADS", str(n))

    # Chỉ định thư mục cache viết được
    cache_dir = os.getenv("WHISPER_CACHE", "/app/.cache/whisper")
    os.makedirs(cache_dir, exist_ok=True)

    # Startup: load model 1 lần
    model = whisper.load_model(
        WHISPER_MODEL,
        device=DEVICE,
        download_root=cache_dir,
        in_memory=False
    )
    if USE_GPU:
        model = model.half()

    app.state.whisper_model = model
    app.state.model_name = f"whisper-{WHISPER_MODEL}"
    app.state.use_fp16 = USE_GPU
    app.state.sem = asyncio.Semaphore(MAX_CONCURRENCY)

    yield

app = FastAPI(title="ASR Service", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
app.include_router(asr_router)

@app.get("/health")
def health():
    return {
        "ok": True,
        "model": getattr(app.state, "model_name", None),
        "gpu": USE_GPU,
        "max_concurrency": MAX_CONCURRENCY
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5005, reload=True)