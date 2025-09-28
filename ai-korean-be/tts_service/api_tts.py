#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import io
import asyncio
from typing import AsyncGenerator, Optional, Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import edge_tts
import base64
import tempfile

router = APIRouter(prefix="/api", tags=["tts"])

# Mặc định bạn có thể đổi cho phù hợp
DEFAULT_FEMALE = "ko-KR-SunHiNeural"
DEFAULT_MALE   = "ko-KR-InJoonNeural"

# Windows asyncio fix
if os.name == "nt":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]
    except Exception:
        pass

# --------- Helpers ----------
def _ensure_signed_percent(v: Optional[str], default="+0%") -> str:
    v = (v or default).strip()
    if v.endswith("st"):  # semitone không được build edge-tts hiện tại chấp nhận ở 1 số bản
        return default
    if v[0] in "+-":
        return v if v.endswith("%") else v + "%"
    return f"+{v}%" if not v.endswith("%") else f"+{v}"

def _ensure_signed_hz(v: Optional[str], default="+0Hz") -> str:
    v = (v or default).strip()
    if v.endswith("st"):  # không hỗ trợ st -> Hz ở đây
        return default
    if v[0] in "+-":
        return v if v.endswith("Hz") else v + "Hz"
    return f"+{v}Hz" if not v.endswith("Hz") else f"+{v}"

async def _pick_voice(gender: Optional[str], short_name: Optional[str]) -> str:
    if short_name:
        return short_name
    want_male = (gender or "").lower() == "male"
    want_female = (gender or "").lower() == "female" or not want_male

    voices = await edge_tts.list_voices()
    shorts = {v.get("ShortName") for v in voices if v.get("ShortName")}
    # Ưu tiên default
    if want_male and DEFAULT_MALE in shorts:
        return DEFAULT_MALE
    if want_female and DEFAULT_FEMALE in shorts:
        return DEFAULT_FEMALE
    # Tìm theo giới tính trong ko-KR
    for v in voices:
        if v.get("Locale") == "ko-KR" and want_male and v.get("Gender") == "Male":
            return v.get("ShortName")
    for v in voices:
        if v.get("Locale") == "ko-KR" and want_female and v.get("Gender") == "Female":
            return v.get("ShortName")
    # Cuối cùng: bất kỳ ko-KR
    for v in voices:
        if v.get("Locale") == "ko-KR":
            return v.get("ShortName")
    raise HTTPException(400, "No Korean voices available at the moment.")

async def _edge_tts_stream(
    text: str, voice: str, rate: str, pitch: str, volume: str
) -> AsyncGenerator[bytes, None]:
    communicate = edge_tts.Communicate(
        text, voice=voice, rate=rate, pitch=pitch, volume=volume
    )
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            yield chunk["data"]

# --------- Schemas ----------
class SynthesizeIn(BaseModel):
    text: str = Field(..., description="Korean text to speak")
    voice: Optional[Literal["male", "female"]] = Field(None, description="Preferred gender (optional if using voice_short_name)")
    voice_short_name: Optional[str] = Field(
        None, description='Explicit voice short name, e.g. "ko-KR-InJoonNeural"'
    )
    rate: Optional[str] = Field(None, description='Percent, e.g., "+0%" / "-10%"')
    pitch: Optional[str] = Field(None, description='Hz, e.g., "+0Hz" / "-20Hz"')
    volume: Optional[str] = Field(None, description='Percent, e.g., "+0%" / "+2%"')

class VoiceInfo(BaseModel):
    short_name: str
    gender: str
    locale: str
    display_name: str

# --------- Routes ----------
@router.get("/voices", response_model=list[VoiceInfo])
async def list_voices(locale: str = Query("ko-KR", description="Filter locale, default ko-KR"), show_all: bool = False):
    voices = await edge_tts.list_voices()
    out = []
    for v in voices:
        if show_all or v.get("Locale") == locale:
            out.append(VoiceInfo(
                short_name=v.get("ShortName", ""),
                gender=v.get("Gender", ""),
                locale=v.get("Locale", ""),
                display_name=v.get("FriendlyName") or v.get("DisplayName") or v.get("ShortName", "")
            ))
    return out

@router.post("/tts")
async def tts_speak(body: SynthesizeIn):
    if not body.text.strip():
        raise HTTPException(400, "text is empty")
    try:
        voice = await _pick_voice(body.voice, body.voice_short_name)
        rate   = _ensure_signed_percent(body.rate, "-10%")
        volume = _ensure_signed_percent(body.volume, "+0%")
        pitch  = _ensure_signed_hz(body.pitch, "+0Hz")
        generator = _edge_tts_stream(body.text, voice, rate, pitch, volume)
        headers = {
            "X-EdgeTTS-Voice": voice,
            "Cache-Control": "no-store",
            # Với browser: set filename
            "Content-Disposition": 'inline; filename="speech.mp3"'
        }
        return StreamingResponse(generator, media_type="audio/mpeg", headers=headers)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"TTS failed: {e}")
    
# Thêm danh sách format thử lần lượt
_EDGE_MP3_FORMATS = [
    "audio-24khz-48kbitrate-mono-mp3",
    "audio-16khz-32kbitrate-mono-mp3",
    "audio-24khz-160kbitrate-mono-mp3",
]

_KR_VOICE_CANDIDATES = [
    "ko-KR-SunHiNeural", "ko-KR-InJoonNeural",
    "ko-KR-JiMinNeural", "ko-KR-SeoHyeonNeural",
    "ko-KR-SoonBokNeural", "ko-KR-YuJinNeural",
    "ko-KR-HyunsuNeural", "ko-KR-BongJinNeural",
    "ko-KR-GookMinNeural", "ko-KR-HyunsuMultilingualNeural",
]

# ===== Edge-only: streaming -> base64 với retry =====
async def _edge_stream_to_base64(text: str, voice: str, rate: str, pitch: str, volume: str, retries: int = 3) -> str:
    """
    Dùng Communicate.stream() để gom audio bytes thay vì save().
    Thử tối đa `retries` lần (có backoff nhẹ). Không fallback gTTS.
    """
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            communicate = edge_tts.Communicate(text, voice=voice, rate=rate, pitch=pitch, volume=volume)
            buf = io.BytesIO()
            got_audio = False
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
                    got_audio = True
            if got_audio and buf.tell() > 0:
                return base64.b64encode(buf.getvalue()).decode("utf-8")
            last_err = RuntimeError("no audio chunks received")
        except Exception as e:
            last_err = e
        # backoff nhẹ để tránh lỗi mạng chớp
        await asyncio.sleep(0.35 * attempt)
    # hết retry mà vẫn fail
    raise last_err if last_err else RuntimeError("unknown edge-tts stream error")

@router.post("/tts_base64")
async def tts_speak_base64(body: SynthesizeIn):
    if not body.text.strip():
        raise HTTPException(400, "text is empty")

    rate   = _ensure_signed_percent(body.rate, "-10%")   # % (ví dụ +0%, -10%)
    volume = _ensure_signed_percent(body.volume, "+0%") # %
    pitch  = _ensure_signed_hz(body.pitch, "+0Hz")      # Hz (ví dụ +0Hz, -20Hz)

    try:
        voice = await _pick_voice(body.voice, body.voice_short_name)
        audio_b64 = await _edge_stream_to_base64(body.text, voice, rate, pitch, volume, retries=3)
        return {
            "engine": "edge-tts",
            "voice": voice,
            "rate": rate, "pitch": pitch, "volume": volume,
            "audio_base64": audio_b64,
            "data_uri": f"data:audio/mpeg;base64,{audio_b64}",
        }
    except Exception as e:
        # KHÔNG fallback gTTS theo yêu cầu -> báo lỗi 502 để client biết Edge fail
        raise HTTPException(status_code=502, detail=f"Edge TTS failed: {e}")