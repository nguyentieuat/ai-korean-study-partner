# api_pronunciationDataset.py
import os, json, base64, mimetypes
from pathlib import Path
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/pronunciation", tags=["pronunciation"])

JSON_DIR = Path(os.getenv("MFA_JSON_DIR", "data/first/json")).resolve()
FIRST_DIR = JSON_DIR.parent.resolve()
AUDIO_DIR = Path(os.getenv("MFA_AUDIO_DIR", FIRST_DIR / "audio")).resolve()

def _to_data_url(file_path: Path) -> str | None:
    if not file_path.exists():
        return None
    mime, _ = mimetypes.guess_type(str(file_path))
    mime = mime or "audio/mpeg"
    b = file_path.read_bytes()
    return f"data:{mime};base64,{base64.b64encode(b).decode('ascii')}"

def _load_and_embed(json_path: Path) -> dict:
    # Trả FULL dữ liệu của 1 level, NHÚNG audio thành base64
    data = json.loads(json_path.read_text(encoding="utf-8"))
    items = data.get("items") or []
    for it in items:
        au = it.get("audioUrl")
        if not au:
            it["audioBase64"] = None
            continue
        audio_path = (FIRST_DIR / au).resolve()
        if not audio_path.exists():
            audio_path = (AUDIO_DIR / Path(au).name).resolve()
        it["audioBase64"] = _to_data_url(audio_path)
        # it.pop("audioUrl", None)  # nếu muốn bỏ hẳn url gốc
    return data

def _load_meta(json_path: Path) -> dict:
    # Chỉ đọc metadata nhẹ: level + focus
    try:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(500, f"Lỗi đọc {json_path.name}: {e}")
    return {
        "level": raw.get("level"),
        "focus": raw.get("focus") or raw.get("title") or "",
        # (tuỳ chọn) "count": len(raw.get("items") or []),
    }

@router.get("/levels")
def get_all_levels():
    """Trả NHẸ: danh sách level + focus (không nhúng audio, không trả items)."""
    if not JSON_DIR.exists():
        raise HTTPException(500, f"Không tìm thấy thư mục JSON: {JSON_DIR}")

    files = sorted(JSON_DIR.glob("level*_pronunciation_data.json"))
    if not files:
        raise HTTPException(404, "Không có file level_*.json")

    metas = [_load_meta(p) for p in files]
    metas.sort(key=lambda d: d.get("level", 0))
    return {"levels": metas}

@router.get("/level/{level}")
def get_one_level(level: int):
    """Trả ĐẦY ĐỦ: 1 level (items + audioBase64) để FE phát ngay."""
    json_path = JSON_DIR / f"level{level}_pronunciation_data.json"
    if not json_path.exists():
        raise HTTPException(404, f"Không có dữ liệu cho level {level}")
    return _load_and_embed(json_path)
