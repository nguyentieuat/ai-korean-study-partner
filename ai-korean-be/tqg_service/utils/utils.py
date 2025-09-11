from gtts import gTTS
from io import BytesIO
from typing import List, Optional
from pydub import AudioSegment
import base64

def text_to_speech(text: str, voice: Optional[int] = 1) -> bytes:
    """
    Tạm thời voice bị bỏ qua vì gTTS không chọn được nam/nữ.
    Trả ra MP3 bytes.
    """
    if not text or not text.strip():
        return b""
    tts = gTTS(text=text, lang="ko", slow=False)
    buf = BytesIO()
    tts.write_to_fp(buf)
    return buf.getvalue()

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