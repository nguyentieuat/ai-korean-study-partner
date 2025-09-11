# api_asrSTT.py
import os, io, time, tempfile, asyncio
from typing import Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from pydantic import BaseModel
from pydub import AudioSegment
import anyio  # chạy blocking func trong thread

router = APIRouter(prefix="/api/stt", tags=["asr"])

class URLPayload(BaseModel):
    url: str
    language: Optional[str] = "ko"

def _to_wav_mono16k(raw_bytes: bytes, in_suffix: str = ".bin") -> bytes:
    import io as _io
    with _io.BytesIO(raw_bytes) as buf:
        audio = AudioSegment.from_file(buf, format=in_suffix.lstrip("."))
        wav = audio.set_channels(1).set_frame_rate(16000)
        out = _io.BytesIO()
        wav.export(out, format="wav")
        return out.getvalue()

def _decode_opts(language: str, use_fp16: bool, fast_mode: bool = True) -> Dict[str, Any]:
    """
    fast_mode=True: greedy, nhanh nhất (gợi ý cho serve).
    Nếu muốn chất lượng cao hơn: set fast_mode=False để bật beam.
    """
    if fast_mode:
        return dict(
            language=language,
            fp16=use_fp16,
            temperature=0.0,
            best_of=1,
            beam_size=None,
            condition_on_previous_text=False,
            # Có thể thêm: compression_ratio_threshold=2.4 để chặn ảo giác dài
        )
    else:
        return dict(
            language=language,
            fp16=use_fp16,
            temperature=0.0,
            best_of=1,
            beam_size=4,  # hoặc 5–8 nếu CPU/GPU chịu được
            patience=None,
            length_penalty=None,
            condition_on_previous_text=False,
        )

def _transcribe_sync(model, path: str, language: str, use_fp16: bool):
    t0 = time.time()
    opts = _decode_opts(language=language, use_fp16=use_fp16, fast_mode=True)
    res = model.transcribe(path, **opts)
    t1 = time.time()
    return {
        "text": res.get("text", "").strip(),
        "language": language,
        "segments": [
            {"start": float(s.get("start", 0.0)),
             "end": float(s.get("end", 0.0)),
             "text": s.get("text","").strip()}
            for s in (res.get("segments") or [])
        ],
        "duration": float(t1 - t0),
    }

@router.post("")
async def transcribe(request: Request, file: UploadFile = File(...), language: str = "ko"):
    model = request.app.state.whisper_model
    use_fp16 = bool(getattr(request.app.state, "use_fp16", False))
    sem: asyncio.Semaphore = request.app.state.sem

    suffix = os.path.splitext(file.filename or "")[1].lower() or ".wav"
    raw = await file.read()
    try:
        wav_bytes = await anyio.to_thread.run_sync(_to_wav_mono16k, raw, suffix)
    except Exception as e:
        raise HTTPException(400, f"audio decode failed: {e}")

    async with sem:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(wav_bytes)
            path = f.name
        try:
            result = await anyio.to_thread.run_sync(_transcribe_sync, model, path, language, use_fp16)
            result["model"] = request.app.state.model_name
            return result
        finally:
            try: os.remove(path)
            except: pass

@router.post("/url")
async def transcribe_url(request: Request, payload: URLPayload):
    import requests
    r = requests.get(payload.url, timeout=30)
    r.raise_for_status()

    ctype = r.headers.get("Content-Type","audio/wav").lower()
    suffix = ".wav"
    if "webm" in ctype: suffix = ".webm"
    elif "mpeg" in ctype or "mp3" in ctype: suffix = ".mp3"
    elif "ogg" in ctype: suffix = ".ogg"

    try:
        wav_bytes = await anyio.to_thread.run_sync(_to_wav_mono16k, r.content, suffix)
    except Exception as e:
        raise HTTPException(400, f"audio decode failed: {e}")

    model = request.app.state.whisper_model
    use_fp16 = bool(getattr(request.app.state, "use_fp16", False))
    sem: asyncio.Semaphore = request.app.state.sem

    async with sem:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(wav_bytes)
            path = f.name
        try:
            result = await anyio.to_thread.run_sync(_transcribe_sync, model, path, payload.language, use_fp16)
            result["model"] = request.app.state.model_name
            return result
        finally:
            try: os.remove(path)
            except: pass