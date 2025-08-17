# speech/transcribe.py
import sys
import os
import torch
import whisper
import tempfile
from werkzeug.datastructures import FileStorage
from pydub import AudioSegment

use_gpu = torch.cuda.is_available()
model = whisper.load_model("medium", device="cuda" if use_gpu else "cpu")

if use_gpu:
    model = model.half()

def transcribe_audio(audio_file: FileStorage):
    try:
        # Ghi dữ liệu vào file tạm
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
            audio_file.stream.seek(0)  # Đảm bảo bắt đầu đọc từ đầu
            temp_audio.write(audio_file.read())  # Ghi nội dung vào file tạm
            temp_audio_path = temp_audio.name
            # Chuẩn hóa audio: mono, 16kHz WAV
            sound = AudioSegment.from_file(temp_audio_path)
            sound = sound.set_channels(1).set_frame_rate(16000)
            sound.export(temp_audio_path, format="wav")

        # Truyền đường dẫn file cho Whisper
        result = model.transcribe(temp_audio_path, language='ko', fp16=use_gpu)
        return result['text']

    except Exception as e:
        # Log lỗi để debug
        print(f"[Transcribe Error] {str(e)}", file=sys.stderr)
        raise RuntimeError("Không thể xử lý audio, vui lòng thử lại.") from e

    finally:
        # Dọn file tạm nếu tồn tại
        if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)