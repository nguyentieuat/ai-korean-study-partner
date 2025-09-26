#api_topikI_generate.py
#api_topikI_generate.py
import base64, json, hashlib, time, os
import re  # <-- thêm
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, List
from contextlib import contextmanager
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from topik1.generate_question import generate_topik_question, warm_topik14_listen_cache, warm_topik56_listen_cache, DataNotFoundError, BadRequestError
from utils.utils import text_to_speech  # <-- bỏ concat_mp3_to_data_uri nếu không dùng

router = APIRouter(prefix="/api/topik", tags=["topik_generate"])

@router.on_event("startup")
async def _startup():
    warm_topik14_listen_cache()
    warm_topik56_listen_cache()

AUDIO_DIR = Path("data/tts/listen")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

_SPK_MALE   = re.compile(r"(남자|^남\b|남성|male)", re.IGNORECASE)
_SPK_FEMALE = re.compile(r"(여자|^여\b|여성|female)", re.IGNORECASE)

def speaker_to_id(speaker: str) -> int:
    s = (speaker or "").strip()
    if _SPK_MALE.search(s):   return 1
    if _SPK_FEMALE.search(s): return 0
    return 1  # default male

def _norm_text(s: str) -> str:
    # gọn khoảng trắng, tránh hash khác nhau do spacing
    return " ".join((s or "").split())

def _dialog_fingerprint(dialog: List[dict], sep_ms: int) -> str:
    """Tạo khóa ổn định từ (voice_id, text), sep_ms và engine/version."""
    # Nếu sau này đổi TTS engine/voice, thêm vào payload để tách file
    payload = {
        "engine": "gTTS",
        "ver": "1",                # bump nếu đổi luật/fmt
        "sep": int(max(0, sep_ms)),
        "dlg": [
            {"v": speaker_to_id(turn.get("speaker") or ""), "t": _norm_text(turn.get("text") or "")}
            for turn in (dialog or [])
            if (turn.get("text") or "").strip()
        ],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

@contextmanager
def _file_lock(lock_path: Path, timeout_sec: int = 10):
    start = time.time()
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if time.time() - start > timeout_sec:
                break
            time.sleep(0.05)
    try:
        yield
    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass

def _concat_segments_to_mp3_bytes(segments: List[bytes], sep_ms: int = 150) -> bytes:
    """Ưu tiên pydub (+ffmpeg); fallback nối bytes MP3 trực tiếp."""
    if not segments:
        return b""
    try:
        from pydub import AudioSegment
        combined = AudioSegment.silent(duration=0)
        sep = AudioSegment.silent(duration=max(0, sep_ms))
        for b in segments:
            if not b: continue
            seg = AudioSegment.from_file(BytesIO(b), format="mp3")
            combined += seg + sep
        buf = BytesIO()
        combined.export(buf, format="mp3", bitrate="192k")
        return buf.getvalue()
    except Exception:
        return b"".join([b for b in segments if b])  # không chèn sep

def _mp3_bytes_to_data_uri(b: bytes) -> str:
    return "data:audio/mpeg;base64," + base64.b64encode(b).decode("ascii")

def build_or_get_listen_mp3_data_uri(dialog: List[dict], sep_ms: int = 150) -> str:
    """
    - Nếu file đã có: đọc bytes -> trả Data URI.
    - Nếu chưa: synth từng lượt (text_to_speech -> MP3 bytes), ghép, lưu file, trả Data URI.
    """
    key = _dialog_fingerprint(dialog, sep_ms)
    mp3_path = AUDIO_DIR / f"listen_{key}.mp3"
    lock_path = AUDIO_DIR / f"listen_{key}.lock"

    if mp3_path.exists():
        return _mp3_bytes_to_data_uri(mp3_path.read_bytes())

    with _file_lock(lock_path):
        if mp3_path.exists():
            return _mp3_bytes_to_data_uri(mp3_path.read_bytes())

        # synth từng lượt
        segments: List[bytes] = []
        for turn in (dialog or []):
            txt = (turn.get("text") or "").strip()
            if not txt: continue
            sid = speaker_to_id(turn.get("speaker") or "")
            seg = text_to_speech(txt, sid)  # -> bytes (MP3)
            if seg: segments.append(seg)

        if not segments:
            return ""  # không có gì để phát

        # ghép & lưu
        mp3_bytes = _concat_segments_to_mp3_bytes(segments, sep_ms=sep_ms)
        try:
            mp3_path.write_bytes(mp3_bytes)  # lưu để tái sử dụng lâu dài
        except Exception:
            pass  # nếu không ghi được vẫn trả về Data URI
        return _mp3_bytes_to_data_uri(mp3_bytes)
    
# ---------- Models ----------
class TopikReq(BaseModel):
    level: str
    category: str
    cau: int
    user_key: Optional[str] = None  

@router.post("/generate")
def api_generate_topik(request: Request, body: TopikReq):
    try:

        print(f"TOPIK generate_question: level={body.level}, category={body.category}, cau={body.cau}") 
        result = generate_topik_question(
            level=body.level, danh_muc=body.category, cau=body.cau, user_key=body.user_key
        )

        dialog = result.get("dialog", []) or []
        question_audio = None  # Data URI

        # Chỉ synth audio cho phần nghe
        cat = (body.category or "").lower()
        if (cat.startswith("listen") or cat.startswith("nghe")) and dialog:
            question_audio = build_or_get_listen_mp3_data_uri(dialog, sep_ms=150)

        return {
            **result,
            "question_audio": question_audio,
            "cau": body.cau
        }
    except BadRequestError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except DataNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

# (tuỳ chọn) reload cache thủ công
@router.post("/reload")
def api_reload():
    try:
        warm_topik14_listen_cache()
        warm_topik56_listen_cache()
        return {"ok": True, "reloaded_at": time.time()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))