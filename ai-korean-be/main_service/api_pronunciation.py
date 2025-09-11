# api_pronunciation.py
import os, json, traceback, re, unicodedata, base64 
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request

router = APIRouter(prefix="/api/pronunciation", tags=["pronunciation"])

MFA_SERVICE_URL = os.getenv("MFA_SERVICE_URL", "http://127.0.0.1:5002").rstrip("/") 

UPLOAD_FOLDER = Path("data/uploads/pronun")
LOG_FOLDER = Path("log/pronun")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)

# ---- Helpers ----
_INVALID_WIN = re.compile(r'[<>:"/\\|?*\x00-\x1F]')
def safe_filename(text: str, max_len: int = 80) -> str:
    # Chuẩn hoá unicode, giữ tiếng Hàn; loại ký tự cấm Windows
    s = unicodedata.normalize("NFKC", text)
    s = _INVALID_WIN.sub("", s)
    # thay các khoảng trắng liên tiếp bằng 1 dấu cách, rồi đổi thành _
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace(" ", "_")
    # tránh tên trống, tránh dấu chấm/space cuối
    if not s:
        s = "utt"
    s = s.strip(" .")
    # cắt độ dài nếu quá dài
    if len(s) > max_len:
        s = s[:max_len].rstrip("._ ")
    if not s:
        s = "utt"
    return s

def choose_ext(upload: UploadFile) -> str:
    # Nếu content-type là wav → .wav; nếu không, dùng suffix gốc; fallback .bin
    ct = (upload.content_type or "").lower()
    if "wav" in ct:
        return ".wav"
    suffix = Path(upload.filename or "").suffix
    return suffix if suffix else ".bin"

@router.get("/levels")
async def levels(req: Request):
    http = req.app.state.http  # httpx.AsyncClient bạn đã gắn ở app startup
    r = await http.get(f"{MFA_SERVICE_URL}/api/pronunciation/levels")
    if r.is_error:
        raise HTTPException(r.status_code, r.text[:500])
    return r.json()

@router.get("/level/{level}")
async def one_level(req: Request, level: int):
    http = req.app.state.http
    r = await http.get(f"{MFA_SERVICE_URL}/api/pronunciation/level/{level}")
    if r.is_error:
        raise HTTPException(r.status_code, r.text[:500])
    return r.json()

@router.post("/evaluate")
async def pronunciation_evaluate(req: Request, text: str = Form(...), audio: UploadFile = File(...)):
    http = req.app.state.http
    try:
        if not text or not audio:
            raise HTTPException(400, "Thiếu text hoặc audio")

        # ---- Gửi sang mfa_service ----
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(400, "File audio rỗng")

        r = await http.post(
            f"{MFA_SERVICE_URL}/api/evaluate-w2v2-forced",
            # f"{MFA_SERVICE_URL}/api/evaluate-mfa-w2v",
            # f"{MFA_SERVICE_URL}/api/evaluate-needleman",
            data={"text": text},
            files={"audio": (audio.filename or "input", audio_bytes, audio.content_type or "application/octet-stream")},
        )
        if r.is_error:
            raise HTTPException(r.status_code, f"Lỗi gọi mfa_service: {r.text[:500]}")
        result = r.json()

        # ---- Lưu file (phục vụ lưu trữ/đối soát) ----
        today_str = datetime.now().strftime("%Y-%m-%d")
        user_folder = UPLOAD_FOLDER / today_str
        user_folder.mkdir(parents=True, exist_ok=True)

        stem = safe_filename(text, max_len=64)
        ext = choose_ext(audio)
        audio_filename = f"{datetime.now().strftime('%H%M%S')}_{stem}{ext}"
        audio_path = user_folder / audio_filename
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        # ---- Log JSONL ----
        with open(LOG_FOLDER / "low_score_log.jsonl", "a", encoding="utf-8") as logf:
            logf.write(json.dumps({
                "text": text,
                "audioFile": audio_path.as_posix(),
                "time": datetime.now().isoformat()
            }, ensure_ascii=False) + "\n")

        # ---- Trả về BASE64 cho client (không trả đường dẫn) ----
        # Suy ra MIME
        mime = (audio.content_type or "").lower()
        if not mime or not mime.startswith("audio/"):
            if ext.lower() == ".wav":
                mime = "audio/wav"
            elif ext.lower() == ".webm":
                mime = "audio/webm"
            else:
                mime = "application/octet-stream"

        b64 = base64.b64encode(audio_bytes).decode("ascii")
        data_url = f"data:{mime};base64,{b64}"

        return {
            "text": text,
            **result,
            "audioMime": mime,
            "audioBase64": b64,
            "audioDataUrl": data_url,
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))