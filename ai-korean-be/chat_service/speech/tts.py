# speech/tts.py
import os
import base64
from io import BytesIO
from typing import Optional, Dict, Any

import requests

TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://127.0.0.1:5004")

def _int_to_gender(v: Optional[int]) -> Optional[str]:
    """
    Map voice int -> gender string theo chuẩn hệ thống:
      1 => 'male'
      0 => 'female'
    If None hoặc khác 0/1 => mặc định 'female'
    """
    if v == 1:
        return "male"
    if v == 0:
        return "female"
    return "female"

def text_to_speech(
    text: str,
    voice: Optional[int] = 1,              # 1=male (default của bạn), 0=female
    *,
    voice_short_name: Optional[str] = None, # ví dụ: "ko-KR-InJoonNeural" / "ko-KR-SunHiNeural"
    rate: Optional[str] = "-10%",             # ví dụ: "-10%" (chậm hơn), "+0%"
    pitch: Optional[str] = None,            # ví dụ: "-20Hz", "+0Hz"
    volume: Optional[str] = None,           # ví dụ: "+0%", "+2%"
    timeout: int = 30
) -> bytes:
    """
    Gọi TTS service (FastAPI) tại /api/tts_base64 và trả về MP3 bytes.
    - Không fallback gTTS.
    - Nếu lỗi hoặc không có audio, trả b"" để caller tự xử lý.
    """
    if not text or not text.strip():
        return b""

    url = TTS_SERVICE_URL.rstrip("/") + "/api/tts_base64"

    payload: Dict[str, Any] = {"text": text}
    if voice_short_name:
        payload["voice_short_name"] = voice_short_name
    else:
        payload["voice"] = _int_to_gender(voice)

    if rate is not None:
        payload["rate"] = rate       # %  e.g. "-10%", "+0%"
    if pitch is not None:
        payload["pitch"] = pitch     # Hz e.g. "-20Hz", "+0Hz"
    if volume is not None:
        payload["volume"] = volume   # %  e.g. "+2%", "+0%"

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        audio_b64 = data.get("audio_base64")
        if not audio_b64:
            return b""
        return base64.b64decode(audio_b64)
    except Exception:
        # Có thể log tại đây nếu cần
        return b""
