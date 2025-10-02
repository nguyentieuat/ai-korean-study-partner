# api_topikI_generate.py
import os, re, json, time, base64, hashlib, unicodedata as _ud
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from contextlib import contextmanager

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

from generate_question import (
    generate_topik_question, warm_all_caches,
    DataNotFoundError, BadRequestError,
)
from utils.utils import text_to_speech  # trả về bytes MP3 (lang=ko)

# ---------------- FastAPI Router ----------------
router = APIRouter(prefix="/api/topik", tags=["topik_generate"])

@router.on_event("startup")
async def _startup():
    warm_all_caches()

# ---------------- Đường dẫn & regex ----------------
AUDIO_DIR = Path("data/tts/listen")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

SFX_ROOT = Path("data/static/audio/sfx")   # thư mục gốc chứa SFX
SFX_ROOT.mkdir(parents=True, exist_ok=True)

# Một số file chuông phổ biến (tùy chọn): đặt sẵn trong sfx/bell/
#   - sfx/bell/딩동댕.mp3
#   - sfx/bell/댕동딩.mp3

# Map nhanh tên SFX (regex) -> relative path (tính từ SFX_ROOT) hoặc path tuyệt đối
_SFX_MAP: Dict[str, str] = {
    r"^딩동댕$|bell|dingdong": "bell/딩동댕.mp3",
    r"^댕동딩$|endbell|end_bell": "bell/댕동딩.mp3",
    r"^따르릉$|phonering|phone_bell": "bell/따르릉.mp3",
    r"짤랑|cash|coin": "ui/짤랑.mp3",
    r"박수|applause": "crowd/박수.mp3",
    r"웃음|laugh": "foley/웃음.mp3",
    r"기침|cough": "foley/기침.mp3",
    r"타자|keyboard": "foley/타자.mp3",
    r"silence\s*([0-9]+)s": "silence/silence_\\1s.mp3",  # ví dụ: (silence 2s) -> silence_2s.mp3
}

# Lượt chỉ có chỉ dẫn phi thoại (không TTS nếu không có SFX)
_NON_SPEECH_HINT = re.compile(
    r"(침묵|정적|무음|속삭임|숨\s*고르기|한숨|기침|웃음|박수|효과음|sfx|silence|pause)",
    re.IGNORECASE,
)

# Bóc từng cặp ngoặc (ngắn gọn để an toàn)
_RX_PARENS = re.compile(r"\(([^)]{1,64})\)")

# Speaker → id (1=nam, 0=nữ)
_SPK_MALE   = re.compile(r"(남자|^남\b|남성|male)", re.IGNORECASE)
_SPK_FEMALE = re.compile(r"(여자|^여\b|여성|female)", re.IGNORECASE)


# ---------------- Tiện ích âm thanh ----------------
def _safe_read(path: Path) -> bytes:
    try:
        if path.exists():
            return path.read_bytes()
    except Exception:
        pass
    return b""

def _concat_segments_to_mp3_bytes(segments: List[bytes], sep_ms: int = 150) -> bytes:
    """Ưu tiên pydub (+ffmpeg) để chèn khoảng lặng; fallback nối mù (không sep)."""
    if not segments:
        return b""
    try:
        from pydub import AudioSegment
        combined = AudioSegment.silent(duration=0)
        sep = AudioSegment.silent(duration=max(0, int(sep_ms)))
        for b in segments:
            if not b:
                continue
            seg = AudioSegment.from_file(BytesIO(b), format="mp3")
            combined += seg + sep
        buf = BytesIO()
        combined.export(buf, format="mp3", bitrate="192k")
        return buf.getvalue()
    except Exception:
        # Không có pydub/ffmpeg → ghép thẳng, không chèn sep
        return b"".join([b for b in segments if b])

def _mp3_bytes_to_data_uri(b: bytes) -> str:
    return "data:audio/mpeg;base64," + base64.b64encode(b).decode("ascii")

def speaker_to_id(speaker: str) -> int:
    s = (speaker or "").strip()
    if _SPK_MALE.search(s):   return 1
    if _SPK_FEMALE.search(s): return 0
    return 1  # mặc định nam

def _norm_text(s: str) -> str:
    return " ".join((s or "").split())

def _slugify_for_fs(s: str) -> str:
    s = _ud.normalize("NFKC", (s or "").strip())
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9A-Za-z가-힣_\-]", "", s)
    return s.lower()

def _lookup_sfx_bytes(tag_text: str) -> bytes:
    """Tìm file SFX theo _SFX_MAP (regex) hoặc theo slug."""
    key = (tag_text or "").strip()
    # 1) map regex (hỗ trợ nhóm số cho silence Xs)
    for rx, rel in _SFX_MAP.items():
        m = re.search(rx, key, flags=re.IGNORECASE)
        if not m:
            continue
        rel_path = rel
        # Nếu có group số, thay \1...\9
        for i in range(1, 10):
            if f"\\{i}" in rel_path and i <= (m.lastindex or 0):
                rel_path = rel_path.replace(f"\\{i}", str(m.group(i)))
        p = (SFX_ROOT / rel_path) if not os.path.isabs(rel_path) else Path(rel_path)
        b = _safe_read(p)
        if b:
            return b
    # 2) tìm theo slug (ưu tiên .mp3)
    slug = _slugify_for_fs(key)
    cand = [
        SFX_ROOT / f"{slug}.mp3",
        SFX_ROOT / slug / f"{slug}.mp3",
        Path("data/static/audio") / f"{slug}.mp3",  # backward-compat
    ]
    for p in cand:
        b = _safe_read(p)
        if b:
            return b
    return b""


# ---------------- Phân tích thoại thành chunk ----------------
Chunk = Tuple[str, object]  # ("tts", str) | ("sfx", bytes) | ("sfx-missing", str)

def _split_text_into_chunks_with_sfx(text: str) -> List[Chunk]:
    """
    Tách text thành các chunk theo thứ tự:
      - ("tts", "nội dung không ngoặc")
      - ("sfx", bytes_mp3) nếu tìm được SFX khớp tag
      - ("sfx-missing", "raw_tag") nếu không tìm được file
    """
    chunks: List[Chunk] = []
    t = text or ""
    idx = 0
    for m in _RX_PARENS.finditer(t):
        pre = t[idx:m.start()]
        if pre.strip():
            chunks.append(("tts", pre.strip()))
        tag_str = m.group(1).strip()
        sfx_bytes = _lookup_sfx_bytes(tag_str)
        if sfx_bytes:
            chunks.append(("sfx", sfx_bytes))
        else:
            chunks.append(("sfx-missing", tag_str))
        idx = m.end()
    tail = t[idx:]
    if tail.strip():
        chunks.append(("tts", tail.strip()))
    return chunks


# ---------------- Fingerprint (ver=2, có dấu vết SFX) ----------------
def _dialog_fingerprint(dialog: List[dict], sep_ms: int) -> str:
    payload = {
        "engine": "gTTS",
        "ver": "2",
        "sep": int(max(0, sep_ms)),
        "dlg": [],
    }
    for turn in (dialog or []):
        spk_id = speaker_to_id(turn.get("speaker") or "")
        raw_txt = _norm_text(turn.get("text") or "")
        chunks = _split_text_into_chunks_with_sfx(raw_txt)
        sig = []
        for kind, val in chunks:
            if kind == "tts":
                sig.append({"k": "t", "v": _norm_text(val)})
            elif kind == "sfx":
                sig.append({"k": "x"})  # không ghi bytes
            elif kind == "sfx-missing":
                sig.append({"k": "m", "v": _slugify_for_fs(val)})
        payload["dlg"].append({"v": spk_id, "sig": sig})
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


# ---------------- File lock đơn giản ----------------
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


# ---------------- Build hoặc lấy Data URI câu nghe ----------------
def build_or_get_listen_mp3_data_uri(dialog: List[dict], sep_ms: int = 150) -> str:
    """
    - Hỗ trợ SFX ở mọi vị trí qua tag ( ... ).
    - Nếu một lượt chỉ toàn chỉ dẫn (và/hoặc SFX), không có text → không TTS; vẫn chèn SFX nếu có.
    - Nếu tag không tìm được file, bỏ qua tag đó (không TTS cho tag).
    """
    key = _dialog_fingerprint(dialog or [], sep_ms)
    mp3_path = AUDIO_DIR / f"listen_{key}.mp3"
    lock_path = AUDIO_DIR / f"listen_{key}.lock"

    if mp3_path.exists():
        return _mp3_bytes_to_data_uri(mp3_path.read_bytes())

    with _file_lock(lock_path):
        if mp3_path.exists():
            return _mp3_bytes_to_data_uri(mp3_path.read_bytes())

        segments: List[bytes] = []
        has_any = False

        for turn in (dialog or []):
            raw_txt = (turn.get("text") or "").strip()
            spk_id = speaker_to_id(turn.get("speaker") or "")
            chunks = _split_text_into_chunks_with_sfx(raw_txt)

            has_tts_text = any(k == "tts" and str(v).strip() for k, v in chunks)
            has_valid_sfx = any(k == "sfx" for k, _ in chunks)

            # Nếu không có gì để phát → bỏ lượt
            if not has_tts_text and not has_valid_sfx:
                # Trường hợp toàn tag thiếu SFX và cũng là chỉ dẫn phi thoại → skip
                continue

            # Xen kẽ SFX và TTS theo thứ tự
            for kind, val in chunks:
                if kind == "sfx" and val:
                    segments.append(val)
                    has_any = True
                elif kind == "tts" and str(val).strip():
                    try:
                        b = text_to_speech(str(val).strip(), spk_id)
                    except Exception:
                        b = b""
                    if b:
                        segments.append(b)
                        has_any = True
                # sfx-missing -> bỏ qua

        if not has_any:
            return ""

        mp3_bytes = _concat_segments_to_mp3_bytes(segments, sep_ms=int(max(0, sep_ms)))
        try:
            mp3_path.write_bytes(mp3_bytes)
        except Exception:
            pass
        return _mp3_bytes_to_data_uri(mp3_bytes)


# ---------------- Schema & API ----------------
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
        question_audio = None

        cat = (body.category or "").lower()
        if (cat.startswith("listen") or cat.startswith("nghe")) and dialog:
            # có thể chỉnh sep_ms theo sở thích
            question_audio = build_or_get_listen_mp3_data_uri(dialog, sep_ms=350)

        return {**result, "question_audio": question_audio, "cau": body.cau}

    except BadRequestError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except DataNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

@router.post("/reload")
def api_reload():
    try:
        warm_all_caches()
        return {"ok": True, "reloaded_at": time.time()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
