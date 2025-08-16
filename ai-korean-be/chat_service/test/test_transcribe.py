
# test/test_transcribe.py
import sys
import os

# Thêm đường dẫn thư mục gốc vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from speech.transcribe import transcribe_audio
from werkzeug.datastructures import FileStorage

def test_transcribe_audio():
    audio_path = "static/uploads/user_fc22440cf33046d685a94bf18476f6c7.webm"
    assert os.path.exists(audio_path), f"File không tồn tại: {audio_path}"

    with open(audio_path, "rb") as f:
        file_storage = FileStorage(
            stream=f,
            filename="user_fc22440cf33046d685a94bf18476f6c7.webm",
            content_type="audio/webm"
        )
        text = transcribe_audio(file_storage)
        print("Transcription:", text)
        assert isinstance(text, str)
        assert len(text.strip()) > 0

if __name__ == "__main__":
    test_transcribe_audio()