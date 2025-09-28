import os, json, base64, requests
from gtts import gTTS
from io import BytesIO
from typing import List, Optional
from pydub import AudioSegment

TTS_SERVICE_URL  = os.getenv("TTS_SERVICE_URL",  "http://127.0.0.1:5004")

# Map INT -> gender theo chuẩn: 1=male, 0=female
def _int_to_gender(v: Optional[int]) -> Optional[str]:
    if v == 1:
        return "male"
    if v == 0:
        return "female"
    # fallback: nếu không truyền, mặc định male
    return "male"

def text_to_speech(
    text: str,
    voice: Optional[int] = 1,          # <-- default male 
    *,
    voice_short_name: Optional[str] = None,
    rate: Optional[str] = "-10%",        # e.g. "-10%"
    pitch: Optional[str] = None,       # e.g. "-20Hz"
    volume: Optional[str] = None,      # e.g. "+0%"
    timeout: int = 30
) -> bytes:
    """
    Gọi TTS service (/api/tts_base64) và trả về MP3 bytes.
    Mapping voice int: 1=male, 0=female.
    voice_short_name (nếu có) sẽ override gender.
    """
    if not text or not text.strip():
        return b""

    url = TTS_SERVICE_URL.rstrip("/") + "/api/tts_base64"
    payload = {"text": text}

    if voice_short_name:
        payload["voice_short_name"] = voice_short_name
    else:
        payload["voice"] = _int_to_gender(voice)

    if rate is not None:
        payload["rate"] = rate
    if pitch is not None:
        payload["pitch"] = pitch
    if volume is not None:
        payload["volume"] = volume

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        audio_b64 = data.get("audio_base64")
        return base64.b64decode(audio_b64) if audio_b64 else b""
    except Exception:
        return b""
    
def concat_mp3_to_base64(segments: List[bytes], sep_ms: int = 150) -> str:
    """
    Nối nhiều MP3 bytes -> 1 MP3 rồi trả về base64 (không lưu file).
    Ưu tiên pydub+ffmpeg; fallback: nối bytes MP3 trực tiếp.
    """
    if not segments:
        return ""

    # Thử pydub (chuẩn nhất)
    try:
        combined = AudioSegment.silent(duration=0)
        sep = AudioSegment.silent(duration=max(0, sep_ms))

        for b in segments:
            if not b:
                continue
            seg = AudioSegment.from_file(BytesIO(b), format="mp3")
            combined += seg + sep

        buf = BytesIO()
        combined.export(buf, format="mp3", bitrate="192k")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        # Fallback: nối bytes MP3 trực tiếp (ít chuẩn hơn nhưng thường vẫn phát được)
        raw = b"".join([b for b in segments if b])
        return base64.b64encode(raw).decode("ascii")

def concat_mp3_to_data_uri(segments: List[bytes], sep_ms: int = 150) -> str:
    """
    Ghép nhiều MP3 bytes -> 1 MP3 rồi trả về Data URI 'data:audio/mpeg;base64,...'.
    Ưu tiên pydub+ffmpeg; fallback: nối bytes MP3 trực tiếp.
    """
    if not segments:
        return ""

    # Thử pydub (chèn khoảng lặng cho tự nhiên)
    try:
        combined = AudioSegment.silent(duration=0)
        sep = AudioSegment.silent(duration=max(0, sep_ms))
        for b in segments:
            if not b:
                continue
            seg = AudioSegment.from_file(BytesIO(b), format="mp3")
            combined += seg + sep
        buf = BytesIO()
        combined.export(buf, format="mp3", bitrate="192k")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:audio/mpeg;base64,{b64}"
    except Exception:
        # Fallback: nối bytes MP3 trực tiếp
        raw = b"".join([b for b in segments if b])
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:audio/mpeg;base64,{b64}"