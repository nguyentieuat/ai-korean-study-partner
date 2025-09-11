# api_cooperate.py
# =================
# Endpoints phục vụ annotation phối hợp & lưu trữ an toàn

# Standard library
import base64
import hashlib
import json
import mimetypes
import os
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

# Third-party
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

router = APIRouter(tags=["cooperate"])

# ---- Nguồn & đích dữ liệu ----
ITEM_SRC = Path("log/pronun/low_score_log.jsonl")      # nguồn item cần annotate
ANNOTATION_LOG = Path("log/pronun/annotations.jsonl")  # log review đếm số lần/hoàn tất

USER_DIR   = Path("cooperate/user")
MARKED_DIR = Path("cooperate/marked/pronun")
ANN_DIR    = Path("cooperate/annotator")
EVENTS_LOG = Path("cooperate/annotator/events.jsonl")

# ======================
# Helpers & Utilities
# ======================

def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def make_item_id(audio_path: str, text: str) -> str:
    """Sinh ID ổn định từ (audio_path, text)."""
    norm = (audio_path.replace("\\", "/").lower().strip() + "|" + (text or "").strip())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]

def _normalize_web_path(p: str) -> str:
    posix = str(p).replace("\\", "/")
    if not posix.startswith("/"):
        posix = "/" + posix
    return posix

def _guess_mime(path: str) -> str:
    m, _ = mimetypes.guess_type(path)
    return m or "audio/wav"

def _to_data_url(b64: str, mime: str) -> str:
    return f"data:{mime};base64,{b64}"

def _atomic_write_json(path: Path, data: Dict[str, Any]):
    """Ghi JSON an toàn: ghi .tmp rồi replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _append_jsonl(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _is_in_dir(path: Path, base_dir: Path) -> bool:
    try:
        return base_dir.resolve() in path.resolve().parents
    except Exception:
        return False

def _stem_without_ext(p: Path) -> str:
    return p.stem

def _resolve_audio_path(p: str) -> Path:
    """
    Chuẩn hoá path:
    - Thử nguyên bản, bỏ leading '/' (web path)
    - Thử thay '/data/upload/' -> '/data/uploads/'
    - Nếu vẫn không thấy: tìm theo basename trong MARKED_DIR
    """
    if not p:
        return Path("")

    s = str(p).replace("\\", "/").strip()
    candidates: List[Path] = []

    if s.startswith("/"):
        candidates.append(Path(s[1:]))  # bỏ leading '/'
    candidates.append(Path(s))

    if "/data/upload/" in s:
        candidates.append(Path(s.replace("/data/upload/", "/data/uploads/")))
        if s.startswith("/"):
            candidates.append(Path(s[1:].replace("data/upload/", "data/uploads/")))

    for c in candidates:
        cp = (Path.cwd() / c).resolve() if not c.is_absolute() else c
        if cp.exists():
            return cp

    # Thử theo basename trong MARKED_DIR
    name = Path(s).name
    cand_marked = MARKED_DIR / name
    if cand_marked.exists():
        return cand_marked.resolve()

    # Không thấy -> trả candidate đầu tiên đã tuyệt đối hoá
    c = candidates[0] if candidates else Path(s)
    return (Path.cwd() / c).resolve() if not c.is_absolute() else c

def _count_reviews_and_finalized(item_id: str) -> tuple[int, bool]:
    """Đếm số review của item_id từ ANNOTATION_LOG; finalized nếu >=5."""
    n = 0
    if ANNOTATION_LOG.exists():
        with ANNOTATION_LOG.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if ev.get("type") == "review" and ev.get("item_id") == item_id:
                    n += 1
    return n, n >= 5

# ======================
# API: danh sách item cần annotate (chưa đủ 5)
# ======================

@router.get("/api/cooperate_annotator")
def cooperate_annotator_from_log(offset: int = 0, limit: int = 10):
    """
    Đọc log nguồn, gom theo item_id (mới nhất), chỉ trả item có review_count < 5.
    Kèm audio_data_url (nếu đọc được file).
    """
    try:
        if not ITEM_SRC.exists():
            raise HTTPException(404, f"Không thấy file log: {ITEM_SRC}")

        # Gom theo item_id, giữ record có time mới nhất
        latest_by_id: Dict[str, Dict[str, Any]] = {}
        with ITEM_SRC.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                text = obj.get("text", "")
                audio_file = obj.get("audioFile", "")
                ts = obj.get("time", "") or ""
                item_id = make_item_id(audio_file, text)

                prev = latest_by_id.get(item_id)
                if not prev or ts >= (prev.get("time") or ""):
                    latest_by_id[item_id] = {
                        "id": item_id,
                        "text": text,
                        "audioFile": audio_file,
                        "time": ts,
                    }

        # Lọc item chưa đủ 5 review
        rows: List[Dict[str, Any]] = []
        for r in latest_by_id.values():
            n_reviews, finalized = _count_reviews_and_finalized(r["id"])
            if finalized:
                continue
            r["_review_count"] = n_reviews
            rows.append(r)

        # Sắp xếp mới nhất trước
        rows.sort(key=lambda r: (r.get("time") or "", r["id"]), reverse=True)

        total = len(rows)
        window = rows[offset : offset + limit]

        data = []
        for r in window:
            audio_file_raw = r["audioFile"]
            abs_p = _resolve_audio_path(audio_file_raw)

            audio_data_url = None
            if abs_p.exists() and abs_p.is_file():
                with abs_p.open("rb") as fp:
                    audio_b64 = base64.b64encode(fp.read()).decode("ascii")
                mime = _guess_mime(abs_p.name)
                audio_data_url = _to_data_url(audio_b64, mime)

            data.append({
                "id": r["id"],
                "text": r["text"],
                "audio_path": _normalize_web_path(audio_file_raw),
                "audio_data_url": audio_data_url,
                "review_count": r.get("_review_count", 0),
                "remaining": max(0, 5 - r.get("_review_count", 0)),
            })

        return {
            "data": data,
            "count": total,
            "offset": offset,
            "limit": limit,
            "returned": len(data),
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Lỗi khi lấy dữ liệu: {e}")

# ======================
# API: lưu annotation (không move lại nếu đã move)
# ======================

@router.post("/api/cooperate_annotator")
async def cooperate_annotator_save(payload: Dict[str, Any]):
    """
    Lưu annotation:
    - Sinh item_id ổn định.
    - Nếu audio chưa ở MARKED_DIR: move vào (lần đầu). Về sau không move nữa.
    - Ghi JSON annotation riêng, log events + review (để đếm 5).
    - KHÔNG chặn >5, nhưng GET sẽ không trả những item đủ 5.
    """
    try:
        annotator_name = payload.get("annotator_name")
        annotator_phone = payload.get("annotator_phone")
        annotations: List[Dict[str, Any]] = payload.get("annotations", []) or []

        if not annotator_phone:
            raise HTTPException(400, "Thiếu annotator_phone")

        saved_json_files: List[str] = []
        last_json_path = ""

        # 1) Cập nhật hồ sơ annotator (đếm tổng)
        USER_DIR.mkdir(parents=True, exist_ok=True)
        user_file = USER_DIR / f"{annotator_phone}.json"

        def _update_user_file():
            new_count = len(annotations)
            total_count = new_count
            if user_file.exists():
                try:
                    old = json.loads(user_file.read_text("utf-8"))
                    total_count = int(old.get("annotation_count", 0)) + new_count
                except Exception:
                    total_count = new_count
            user_info = {
                "name": annotator_name,
                "phone": annotator_phone,
                "annotation_count": total_count,
                "last_submit": _iso_now(),
            }
            _atomic_write_json(user_file, user_info)
            return user_info

        user_info = await run_in_threadpool(_update_user_file)

        # 2) Xử lý từng annotation
        for ann in annotations:
            try:
                raw_audio_path = ann.get("audio_path") or ann.get("audioFile") or ""
                text_for_id = ann.get("text") or ann.get("target_text") or ""
                item_id = ann.get("id") or make_item_id(raw_audio_path, text_for_id)

                # Đếm review hiện tại
                n_reviews, _ = _count_reviews_and_finalized(item_id)

                # Resolve path & move nếu cần (chỉ khi chưa ở MARKED_DIR)
                src_path = _resolve_audio_path(raw_audio_path)
                dst_path = src_path
                if src_path and src_path.exists() and src_path.is_file() and not _is_in_dir(src_path, MARKED_DIR):
                    MARKED_DIR.mkdir(parents=True, exist_ok=True)
                    dst_path = MARKED_DIR / src_path.name
                    if not dst_path.exists():
                        def _move():
                            try:
                                shutil.move(str(src_path), str(dst_path))
                            except Exception:
                                shutil.copy2(str(src_path), str(dst_path))
                                try:
                                    os.unlink(str(src_path))
                                except Exception:
                                    pass
                        await run_in_threadpool(_move)
                    else:
                        dst_path = dst_path.resolve()

                audio_path_norm = str(dst_path).replace("\\", "/")

                ann_record = {
                    **ann,
                    "item_id": item_id,
                    "annotator": {"name": annotator_name, "phone": annotator_phone},
                    "submitted_at": _iso_now(),
                    "id": uuid4().hex,   # id của bản annotation
                    "audio_path": audio_path_norm,
                }

                stem = _stem_without_ext(dst_path) if dst_path else uuid4().hex
                ann_json_path = ANN_DIR / f"{stem}_{item_id}_{int(time.time())}.json"
                await run_in_threadpool(_atomic_write_json, ann_json_path, ann_record)

                # Audit event
                event = {
                    "type": "annotation_saved",
                    "ts": _iso_now(),
                    "annotator_phone": annotator_phone,
                    "annotator_name": annotator_name,
                    "json_path": str(ann_json_path).replace("\\", "/"),
                    "audio_path": audio_path_norm,
                    "item_id": item_id,
                    "annotation_id": ann_record["id"],
                }
                await run_in_threadpool(_append_jsonl, EVENTS_LOG, event)

                # Log review (đếm +1)
                new_count = n_reviews + 1
                review_ev = {
                    "type": "review",
                    "ts": _iso_now(),
                    "item_id": item_id,
                    "annotator_phone": annotator_phone,
                    "json_path": str(ann_json_path).replace("\\", "/"),
                    "count_after": new_count,
                    "finalize": new_count >= 5,
                }
                await run_in_threadpool(_append_jsonl, ANNOTATION_LOG, review_ev)

                saved_json_files.append(str(ann_json_path).replace("\\", "/"))
                last_json_path = str(ann_json_path).replace("\\", "/")

            except HTTPException:
                raise
            except Exception as e:
                traceback.print_exc()
                err_event = {
                    "type": "annotation_error",
                    "ts": _iso_now(),
                    "annotator_phone": annotator_phone,
                    "error": str(e),
                    "raw": ann,
                }
                await run_in_threadpool(_append_jsonl, EVENTS_LOG, err_event)

        return {
            "message": "Lưu thành công!",
            "json_file": last_json_path,
            "json_files": saved_json_files,
            "user": user_info,
            "saved_count": len(saved_json_files),
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Lỗi xử lý: {e}")

# ======================
# API: lưu TOPIK annotator (FastAPI thuần)
# ======================

@router.post("/api/cooperate_topik_annotator")
async def cooperate_topik_annotator_save(request: Request):
    """
    Lưu các annotation TOPIK (JSON thuần từ client).
    """
    try:
        data = await request.json()
        annotator_name = data.get("annotator_name")
        annotator_phone = data.get("annotator_phone")
        annotations = data.get("annotations", []) or []

        if not annotator_phone:
            raise HTTPException(400, "Thiếu annotator_phone")

        USER_DIR.mkdir(parents=True, exist_ok=True)
        user_file = USER_DIR / f"{annotator_phone}.json"

        # update user count
        new_count = len(annotations)
        total_count = new_count
        if user_file.exists():
            try:
                old_data = json.loads(user_file.read_text("utf-8"))
                total_count = int(old_data.get("annotation_count", 0)) + new_count
            except Exception:
                total_count = new_count

        user_info = {
            "name": annotator_name,
            "phone": annotator_phone,
            "annotation_count": total_count,
            "last_submit": _iso_now(),
        }
        _atomic_write_json(user_file, user_info)

        saved_files = []
        topik_dir = Path("cooperate/topik_annotator")
        topik_dir.mkdir(parents=True, exist_ok=True)

        for ann in annotations:
            try:
                ts = datetime.now().strftime("%Y%m%d%H%M%S")
                fname = f"{ts}_{uuid4().hex[:8]}.json"
                ann_file = topik_dir / fname
                _atomic_write_json(ann_file, ann)
                saved_files.append(str(ann_file).replace("\\", "/"))
            except Exception as e:
                traceback.print_exc()
                break

        return {"message": "Lưu thành công!", "json_files": saved_files, "user": user_info}

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Lỗi xử lý: {e}")

# ======================
# API: lưu Vitspre (multipart/form-data)
# ======================

@router.post("/api/cooperate_vitspre")
async def cooperate_vitspre_save(
    request: Request,
    annotator_name: Optional[str] = Form(None),
    annotator_phone: str = Form(...),
    records: str = Form("[]"),
    files: List[UploadFile] = File(None),  # không dùng trực tiếp; lấy từng "audio_{i}" từ form
):
    try:
        recs = json.loads(records or "[]")
        new_count = len(recs)
        json_path = ""

        if new_count > 0:
            # update user info
            USER_DIR.mkdir(parents=True, exist_ok=True)
            user_file = USER_DIR / f"{annotator_phone}.json"

            total_count = new_count
            if user_file.exists():
                try:
                    old = json.loads(user_file.read_text("utf-8"))
                    total_count = int(old.get("annotation_count", 0)) + new_count
                except Exception:
                    total_count = new_count

            user_info = {
                "name": annotator_name or "",
                "phone": annotator_phone,
                "annotation_count": total_count,
                "last_submit": _iso_now(),
            }
            _atomic_write_json(user_file, user_info)

            # save audio_i
            saved_records = []
            audio_dir = Path("cooperate/vitspre/audio")
            audio_dir.mkdir(parents=True, exist_ok=True)

            form = await request.form()
            for i, record in enumerate(recs):
                file = form.get(f"audio_{i}")
                if file and hasattr(file, "file"):
                    fname = f"{uuid4().hex}_{(record.get('text_clean','') or '')}.wav"
                    path = audio_dir / fname
                    with path.open("wb") as f:
                        f.write(file.file.read())
                    record["audio_path"] = str(path).replace("\\", "/")
                saved_records.append(record)

            # save metadata json
            vitspre_dir = Path("cooperate/vitspre")
            vitspre_dir.mkdir(parents=True, exist_ok=True)
            text_key = recs[0].get("text_clean", "vitspre") if recs else "vitspre"
            json_path = vitspre_dir / f"vitspre_{int(time.time())}_{text_key}.json"
            _atomic_write_json(json_path, saved_records)

        return {"message": "Lưu thành công!", "json_file": str(json_path).replace("\\", "/")}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Lỗi lưu Vitspre: {e}")
