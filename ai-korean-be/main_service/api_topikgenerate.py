# api_topikgenerate.py
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import asyncio
import hashlib
import json
import os
import traceback

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

router = APIRouter(tags=["topik"])

CHAT_SERVICE_URL = os.getenv("CHAT_SERVICE_URL", "http://127.0.0.1:5001")
TQG_SERVICE_URL  = os.getenv("TQG_SERVICE_URL",  "http://127.0.0.1:5003")

class TopikReq(BaseModel):
    level: str
    category: str
    cau: int

@router.post("/api/generate_question")
async def topik_generate_question(req: Request, body: TopikReq):
    http = req.app.state.http
    try:
        user_key = req.headers.get("X-User-Id") or f"ip:{req.client.host}"
        # Gọi TQG
        r = await http.post(f"{TQG_SERVICE_URL}/api/topik/generate",
                            json={"level": body.level.lower(),
                                  "category": body.category,
                                  "cau": body.cau,
                                  "user_key": user_key})
        if r.is_error:
            raise HTTPException(r.status_code, f"TQG: {r.text[:500]}")
        result = r.json()
        return result
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Lỗi gọi TQG Service: {e}")

LOG_DIR = Path("log/feedback")
SNAP_DIR = LOG_DIR / "snapshots"
IDEM_DIR = LOG_DIR / "idem"
for d in (LOG_DIR, SNAP_DIR, IDEM_DIR):
    d.mkdir(parents=True, exist_ok=True)

def _canon_json(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _write_append(path: Path, line: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line)

async def _append_jsonl_atomic(path: Path, record: dict, timeout_sec: int = 10):
    lock_path = path.with_suffix(path.suffix + ".lock")
    deadline = asyncio.get_event_loop().time() + timeout_sec
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if asyncio.get_event_loop().time() > deadline:
                break
            await asyncio.sleep(0.05)
    try:
        line = _canon_json(record) + "\n"
        await run_in_threadpool(_write_append, path, line)
    finally:
        try: os.remove(lock_path)
        except FileNotFoundError: pass

def _today_log_path() -> Path:
    d = datetime.now(timezone.utc).date().isoformat()
    return LOG_DIR / f"{d}.jsonl"

def _save_snapshot_once(snapshot: dict) -> str:
    # chuẩn hoá snapshot tối thiểu; KHÔNG chứa audio/base64
    minimal = {
        "id": snapshot.get("id"),
        "title": snapshot.get("title"),
        "question": snapshot.get("question"),
        "dialog": snapshot.get("dialog") or [],
        "choices": snapshot.get("choices") or {},
        "answer": snapshot.get("answer"),
        "score": snapshot.get("score"),
        "level": snapshot.get("level"),
        "type": snapshot.get("type"),
        "cau": snapshot.get("cau"),
        "schema": 1,
    }
    raw = _canon_json(minimal)
    h = _sha1(raw)
    p = SNAP_DIR / f"{h}.json"
    if not p.exists():
        p.write_text(raw, encoding="utf-8")
    return h

def _idem_path(key: str) -> Path:
    safe = _sha1(key)
    return IDEM_DIR / f"{safe}.txt"

@router.post("/api/feedback")
async def topik_feedback_save(req: Request):
    try:
        client_ip = req.client.host if req.client else None
        user_key = req.headers.get("X-User-Id") or (f"ip:{client_ip}" if client_ip else "anon")
        idem_key = req.headers.get("Idempotency-Key") or req.headers.get("X-Idempotency-Key")
        user_agent = req.headers.get("User-Agent")

        try:
            body = await req.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Payload must be JSON")

        # Build snapshot từ client (ưu tiên question_snapshot), fallback từ các trường rời
        snap_src = body.get("question_snapshot") or {
            "id": body.get("questionId") or body.get("question_id"),
            "title": body.get("title"),
            "question": body.get("question"),
            "dialog": body.get("dialog"),
            "choices": body.get("choices"),
            "answer": body.get("answer"),
            "score": body.get("score"),
            "level": body.get("level"),
            "type": body.get("type"),
            "cau": body.get("cau"),
        }
        question_hash = _save_snapshot_once(snap_src)

        # Idempotency (tùy chọn)
        prior_event_id = None
        if idem_key:
            idem_file = _idem_path(idem_key)
            if idem_file.exists():
                try:
                    prior_event_id = idem_file.read_text(encoding="utf-8").strip()
                except Exception:
                    prior_event_id = None
            if prior_event_id:
                return {"ok": True, "event_id": prior_event_id, "dedup": True}

        event_id = str(uuid4())
        event = {
            "schema": 2,
            "event_id": event_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "user_key": user_key,
            "client_ip": client_ip,
            "user_agent": user_agent,
            "idempotency_key": idem_key,

            # fields để phân tích nhanh
            "level": body.get("level"),
            "type": body.get("type"),
            "cau": body.get("cau"),
            "question_id": body.get("questionId") or body.get("question_id"),
            "question_hash": question_hash,

            "reaction": body.get("reaction"),
            "reason_tags": body.get("reason_tags"),
            "free_text": body.get("free_text"),

            "answer_selected": body.get("answer_selected"),
            "is_correct": body.get("is_correct"),
            "timer_seconds_left": body.get("timer_seconds_left"),
            "time_limit": body.get("time_limit") or 120,
        }

        # Ghi event
        log_path = _today_log_path()
        await _append_jsonl_atomic(log_path, event)

        # Lưu mapping idem -> event_id (nếu có)
        if idem_key:
            try:
                _idem_path(idem_key).write_text(event_id, encoding="utf-8")
            except Exception:
                pass

        return {"ok": True, "event_id": event_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"log error: {e}")