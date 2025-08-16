from gtts import gTTS
import os
import uuid
from io import BytesIO

def text_to_speech(text: str):
    try:
        # Tạo tệp âm thanh từ văn bản (ngôn ngữ Hàn)
        tts = gTTS(text=text, lang='ko')
        
        # filename = f"reply_{uuid.uuid4().hex}.mp3"
        # path = os.path.join("static", "tts", filename)
        # os.makedirs(os.path.dirname(path), exist_ok=True)

        # tts.save(path)
        # url = f"/static/tts/{filename}"
        audio_io = BytesIO()
        tts.write_to_fp(audio_io)   # ghi TTS vào BytesIO
        audio_io.seek(0)             # reset con trỏ
        return audio_io

    except Exception as e:
        print("❌ Lỗi text_to_speech:", e)
        return None, None
