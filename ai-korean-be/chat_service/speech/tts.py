# speech/tts.py
from gtts import gTTS
from io import BytesIO
from typing import Optional

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
