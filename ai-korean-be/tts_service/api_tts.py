# api_tts.py
import os
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Engine bạn đã có sẵn
from tts_engine_edge import (
    synthesize, MODEL_KOR, CACHE_DIR, DEVICE_ID, SAMPLE_RATE, VOICE_MAP
)

router = APIRouter(prefix="/api", tags=["tts"])

# ====== Schemas ======
class TTSRequest(BaseModel):
    text: str = Field(..., description="Nội dung cần đọc")
    slow_factor: float = Field(0.9, ge=0.2, le=2.0, description="0.5..1.5 (map -> rate %)")
    pause_ms: int = Field(400, ge=0, le=5000, description="Khoảng nghỉ giữa câu (ms)")
    cache: bool = Field(True, description="Bật/tắt cache")
    speaker: Literal["m", "f"] = Field("f", description="Giọng m/f")


# ====== Routes ======
@router.get("/voices")
def voices():
    """Trả về map voice hiện tại cho m/f."""
    return {"ok": True, "voices": VOICE_MAP}


@router.post("/tts")
def tts(req: TTSRequest):
    """Tạo TTS, trả metadata + cache_key (file tải qua /api/tts/file)."""
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text rỗng.")

    try:
        wav_bytes, sr, key = synthesize(
            text=text,
            slow_factor=req.slow_factor,
            pause_ms=req.pause_ms,
            use_cache=req.cache,
            speaker=req.speaker,
        )
        return {
            "ok": True,
            "sampling_rate": sr,
            "cache_key": key,
            "bytes": len(wav_bytes),
            "speaker": req.speaker,
            "engine": "edge-tts",
            "model": MODEL_KOR,
            "device": "cuda" if DEVICE_ID >= 0 else "cpu",
            "sample_rate": SAMPLE_RATE,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/tts/file")
def tts_file(
    key: str = Query(..., description="cache_key trả về từ /api/tts"),
    filename: str = Query("tts.wav", description="Tên file tải về"),
):
    """Tải file WAV từ cache theo cache_key."""
    path = os.path.join(CACHE_DIR, f"{key}.wav")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="not found")

    return FileResponse(path, media_type="audio/wav", filename=filename)
