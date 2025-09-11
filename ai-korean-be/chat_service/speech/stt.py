# speech/stt.py
import os
import io
import json
import traceback
from typing import Optional
import requests
from werkzeug.datastructures import FileStorage

# ====== Cấu hình ======
# Base URL service ASR (FastAPI)
# Ví dụ: http://localhost:5005  → endpoint thực tế: {ASR_API_BASE}/api/stt/
ASR_API_BASE: str = os.getenv("ASR_API_BASE", "http://localhost:5005").rstrip("/")
ASR_ENDPOINT: str = ASR_API_BASE + "/api/stt/"  # chú ý dấu "/" cuối để tránh 307
ASR_LANGUAGE_DEFAULT: str = os.getenv("ASR_LANGUAGE", "ko")

# Timeout (connect, read) giây
CONNECT_TIMEOUT = float(os.getenv("ASR_CONNECT_TIMEOUT", "5"))
READ_TIMEOUT = float(os.getenv("ASR_READ_TIMEOUT", "180"))


def transcribe_audio(
    audio_file: FileStorage,
    language: Optional[str] = None,
) -> str:
    """
    Gửi file audio lên ASR service và trả về text.
    - audio_file: Flask FileStorage (multipart)
    - language: mã ngôn ngữ (mặc định lấy từ env 'ASR_LANGUAGE', fallback 'ko')

    Trả về: transcript text (str).
    Ném RuntimeError nếu lỗi (để caller xử lý).
    """
    lang = (language or ASR_LANGUAGE_DEFAULT) or "ko"

    # Đảm bảo stream ở đầu & chuẩn bị multipart
    audio_file.stream.seek(0)

    filename = audio_file.filename or "audio.bin"
    content_type = audio_file.mimetype or "application/octet-stream"

    files = {
        # (fieldname, (filename, fileobj/bytes, content_type))
        "file": (filename, audio_file.stream, content_type),
    }
    data = {"language": lang}

    try:
        resp = requests.post(
            ASR_ENDPOINT,
            files=files,
            data=data,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            allow_redirects=True,  # dự phòng nếu server redirect /api/stt <-> /api/stt/
        )
    except Exception as e:
        # Lỗi kết nối network
        print(f"[ASR] Request error: {e}")
        traceback.print_exc()
        raise RuntimeError("Không thể kết nối ASR service. Vui lòng thử lại sau.") from e

    # HTTP error (4xx/5xx)
    if not resp.ok:
        # cố thử đọc message JSON từ server
        try:
            err_payload = resp.json()
        except Exception:
            err_payload = {"detail": resp.text.strip()[:500]}
        msg = err_payload.get("detail") if isinstance(err_payload, dict) else str(err_payload)
        raise RuntimeError(f"ASR service trả lỗi {resp.status_code}: {msg}")

    # Parse JSON hợp lệ
    try:
        payload = resp.json()
    except json.JSONDecodeError:
        raise RuntimeError("ASR service trả về dữ liệu không phải JSON.")

    # Lấy text
    text = (payload.get("text") or "").strip()
    if not text:
        # Cho hiển thị payload rút gọn để debug nhanh
        short_payload = json.dumps({k: payload.get(k) for k in ("text", "language", "model")}, ensure_ascii=False)
        raise RuntimeError(f"ASR service không trả text hợp lệ. Payload: {short_payload}")

    return text
