import subprocess
import os
import sys
import uuid

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
