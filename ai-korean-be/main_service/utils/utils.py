import subprocess
import os, base64, requests
import sys
import uuid
from typing import List, Optional

def convert_webm_to_mp3(webm_path):
    mp3_path = webm_path.replace(".webm", ".mp3")
    subprocess.run([
        "ffmpeg", "-i", webm_path,
        "-vn", "-ar", "44100", "-ac", "2", "-b:a", "192k",
        mp3_path
    ])
    return mp3_path


def save_audio_upload(audio_bytes, file_type):
    try:
        upload_dir = os.path.join("static", "uploads", file_type)
        os.makedirs(upload_dir, exist_ok=True)

        audio_filename = f"{file_type}_{uuid.uuid4().hex}.webm"
        audio_path = os.path.join(upload_dir, audio_filename)

        if not audio_bytes:
            raise ValueError("File audio rỗng hoặc không đọc được dữ liệu.")

        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        audio_url_goc = f"/static/uploads/{file_type}/{audio_filename}"
        return audio_url_goc

    except Exception as e:
        print(f"[Save Audio Error] {str(e)}", file=sys.stderr)
        raise RuntimeError("Không thể lưu file audio.") from e


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
    voice: Optional[int] = 1,
    *,
    voice_short_name: Optional[str] = None,
    rate: Optional[str] = "-10%",
    pitch: Optional[str] = None,
    volume: Optional[str] = None,
    timeout: int = 30,
) -> str:
    """
    Gọi TTS service (/api/tts_base64) và trả về CHUỖI base64 (không decode).
    Mapping voice int: 1=male, 0=female.
    voice_short_name (nếu có) sẽ override gender.
    """
    if not text or not text.strip():
        return ""

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
        audio_b64 = data.get("audio_base64") or ""
        # Trả về NGUYÊN CHUỖI base64
        return audio_b64
    except Exception:
        return ""