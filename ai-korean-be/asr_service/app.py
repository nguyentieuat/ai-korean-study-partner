# app.py
import os, asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import torch, whisper

# ---- FIX: distinct router aliases ----
from routers.api_asrSTT import router as asr_http
from routers.asr_ws import router as asr_ws

# ---- CTC/W2V2 preload ----
from core.ctc_gate import preload_ctc, ctc_dummy_forward, set_ctc_model_name, set_ctc_device
from core.w2v2_forced_scoring import preload_forced_encoder

# ==== Detect device ====
USE_GPU = torch.cuda.is_available()
DEVICE  = "cuda" if USE_GPU else "cpu"

def _default_model_for_device() -> str:
    env = os.getenv("WHISPER_MODEL")
    if env:
        return env
    if USE_GPU:
        return "small"
    else:
        return "base"
WHISPER_MODEL = _default_model_for_device()

def _default_concurrency() -> int:
    env = os.getenv("MAX_CONCURRENCY")
    if env: return int(env)
    if USE_GPU:
        return 1
    else:
        cores = os.cpu_count() or 4
        return max(1, min(6, cores // 2))
MAX_CONCURRENCY = _default_concurrency()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Threads & math modes
    if USE_GPU:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    else:
        n = max(1, min((os.cpu_count() or 4), MAX_CONCURRENCY * 2))
        try:
            torch.set_num_threads(n)
        except Exception:
            pass
        os.environ.setdefault("OMP_NUM_THREADS", str(n))
        os.environ.setdefault("MKL_NUM_THREADS", str(n))

    # Cache dirs
    hf_cache = os.getenv("HF_HOME", "/app/.cache/huggingface")
    os.makedirs(hf_cache, exist_ok=True)
    os.environ.setdefault("HF_HOME", hf_cache)

    whisper_cache = os.getenv("WHISPER_CACHE", "/app/.cache/whisper")
    os.makedirs(whisper_cache, exist_ok=True)

    # ==== 1) Load Whisper ====
    model = whisper.load_model(
        WHISPER_MODEL,
        device=DEVICE,
        download_root=whisper_cache,
        in_memory=False
    )
    if USE_GPU:
        model = model.half()
    app.state.whisper_model = model
    app.state.model_name = f"whisper-{WHISPER_MODEL}"
    app.state.use_fp16 = USE_GPU
    app.state.sem = asyncio.Semaphore(MAX_CONCURRENCY)

    # ==== 2) Preload W2V2 CTC (ctc_gate) ====
    # đọc ENV W2V_MODEL nếu có, vd: "Kkonjeong/wav2vec2-base-korean"
    w2v_model_name = os.getenv("W2V_MODEL", "Kkonjeong/wav2vec2-base-korean")  # None => dùng default trong ctc_gate.py
    # set optional overridesz
    if w2v_model_name:
        set_ctc_model_name(w2v_model_name)
    set_ctc_device(DEVICE)
    preload_ctc(model_name=w2v_model_name, device=DEVICE)
    # warm-up 1 forward silence
    ctc_dummy_forward(sample_rate=16000, sec=0.6)

    # ==== 3) Preload Forced-Encoder (w2v2_forced_scoring) ====
    enc_model_name = os.getenv("W2V_ENCODER_MODEL", "Kkonjeong/wav2vec2-base-korean")
    preload_forced_encoder(device=DEVICE, model_name=enc_model_name)

    yield

app = FastAPI(title="ASR Service", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---- FIX: register distinct routers ----
app.include_router(asr_http)
app.include_router(asr_ws)

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
