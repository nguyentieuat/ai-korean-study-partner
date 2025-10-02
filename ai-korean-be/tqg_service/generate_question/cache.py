# generate_question/cache.py
from __future__ import annotations

import os                     
import json
import time
import random
import threading
import base64
import mimetypes
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Iterable

from .utils import stable_id, norm

# ====== INLINE ẢNH (Base64) ======
INLINE_IMAGES = (os.getenv("INLINE_IMAGES", "1") == "1")
_B64_IMG_CACHE: Dict[tuple, str] = {}

_CACHE_LOCK = threading.RLock()

# Cache theo thư mục:
#  - by_qno:   dynamic {qno: [records]}
#  - by_score: {2:[...],3:[...],4:[...]}
#  - by_group: {parent_id: {qno: rec}}
_DIR_CACHE: Dict[Path, dict] = defaultdict(lambda: {
    "loaded_at": 0.0,
    "files": [],
    "mtimes": {},
    "by_qno": defaultdict(list),
    "by_score": {2: [], 3: [], 4: []},
    "by_group": defaultdict(dict),
})

_RECENT_BY_USER: Dict[tuple, deque] = defaultdict(lambda: deque(maxlen=200))


def _is_rel_image_path(v: str) -> bool:
    if not isinstance(v, str):
        return False
    s = v.strip().lower()
    return (
        (s.startswith("images/") or s.startswith("img/") or s.startswith("icons/"))
        and s.rsplit(".", 1)[-1] in {"png","jpg","jpeg","webp","gif","svg"}
    )

def _file_to_data_uri(full: Path) -> str:
    mime, _ = mimetypes.guess_type(str(full))
    b = full.read_bytes()
    return f"data:{mime or 'application/octet-stream'};base64,{base64.b64encode(b).decode()}"

def _inline_one_image(v: str, base_dir_for_images: Path) -> str:
    if not _is_rel_image_path(v):
        return v
    full = (base_dir_for_images / v).resolve()
    if not str(full).startswith(str(base_dir_for_images.resolve())) or not full.is_file():
        return v
    key = (str(base_dir_for_images), v)
    if key in _B64_IMG_CACHE:
        return _B64_IMG_CACHE[key]
    data_uri = _file_to_data_uri(full)
    _B64_IMG_CACHE[key] = data_uri
    return data_uri

def _inline_images_in_obj_fields(sub: dict, base_dir_for_images: Path) -> None:
    if not INLINE_IMAGES:
        return
    if isinstance(sub.get("passage"), str):
        sub["passage"] = _inline_one_image(sub["passage"], base_dir_for_images)
    opts = sub.get("options")
    if isinstance(opts, dict):
        new_opts = {}
        for k, v in opts.items():
            new_opts[k] = _inline_one_image(v, base_dir_for_images) if isinstance(v, str) else v
        sub["options"] = new_opts

def _load_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

def _expand_multi_question(obj: dict) -> Iterable[dict]:
    qs = obj.get("questions")
    if not isinstance(qs, list) or not qs:
        yield obj
        return
    parent_id = obj.get("id") or stable_id(obj)
    shared = {k: obj.get(k) for k in [
        "type", "section", "level", "title", "dialogue", "passage_ko", "passage"
    ]}
    for q in qs:
        if not isinstance(q, dict):
            continue
        try:
            qno = int(q.get("question_no"))
        except Exception:
            qno = None
        opts = q.get("options")
        if isinstance(opts, dict):
            opts = {k: (v.strip() if isinstance(v, str) else v) for k, v in opts.items()}
        elif isinstance(opts, list):
            opts = [v.strip() for v in opts if isinstance(v, str)]
        rec = {
            **shared,
            "parent_id": parent_id,
            "question_no": qno,
            "question": q.get("question") or q.get("prompt"),
            "options": opts,
            "choices": q.get("choices"),
            "answer": q.get("answer"),
            "score": q.get("score"),
            "explanation": q.get("explanation", ""),
        }
        yield rec

def _needs_refresh(dir_path: Path) -> bool:
    cache = _DIR_CACHE[dir_path]
    if not cache["files"]:
        return True
    current = list(dir_path.glob("*.jsonl"))
    if set(current) != set(cache["files"]):
        return True
    for p in current:
        m = p.stat().st_mtime
        if cache["mtimes"].get(p) != m:
            return True
    return False

def refresh_cache(dir_path: Path) -> None:
    with _CACHE_LOCK:
        by_qno   = defaultdict(list)
        by_score = {2: [], 3: [], 4: []}
        by_group = defaultdict(dict)

        files  = list(dir_path.glob("*.jsonl"))
        mtimes = {p: p.stat().st_mtime for p in files}

        for p in files:
            for obj in _load_jsonl(p):
                for sub in _expand_multi_question(obj):
                    _inline_images_in_obj_fields(sub, dir_path.parent / "dataset")
                    qid = sub.get("id") or stable_id(sub)
                    rec = {"id": qid, "data": sub}

                    qno = sub.get("question_no")
                    try:
                        qno = int(qno) if qno is not None else None
                    except Exception:
                        qno = None
                    if qno is not None:
                        by_qno[qno].append(rec)

                    sc = sub.get("score")
                    try:
                        sc = int(sc) if sc is not None else None
                    except Exception:
                        sc = None
                    if sc in (2, 3, 4):
                        by_score[sc].append(rec)

                    pid = sub.get("parent_id")
                    if pid and qno is not None:
                        by_group[pid][qno] = rec

        for k in list(by_qno.keys()):
            random.shuffle(by_qno[k])
        for k in list(by_score.keys()):
            random.shuffle(by_score[k])

        _DIR_CACHE[dir_path].update({
            "by_qno": by_qno,
            "by_score": by_score,
            "by_group": by_group,
            "files": files,
            "mtimes": mtimes,
            "loaded_at": time.time(),
        })

def ensure_ready(dir_path: Path) -> None:
    if not dir_path.exists():
        raise FileNotFoundError(f"Không thấy thư mục: {dir_path}")
    if _needs_refresh(dir_path):
        refresh_cache(dir_path)

def choose_nonrepeating(pool: List[dict], hist_key: tuple) -> dict:
    if not pool:
        raise ValueError("Pool rỗng.")
    with _CACHE_LOCK:
        recent_ids = set(_RECENT_BY_USER[hist_key])
        candidates = [q for q in pool if q["id"] not in recent_ids]
        if not candidates:
            _RECENT_BY_USER[hist_key].clear()
            candidates = pool
        choice = random.choice(candidates)
        _RECENT_BY_USER[hist_key].append(choice["id"])
        return choice

def choose_pair_by_group(cache: dict, qno_a: int, qno_b: int, hist_key: tuple) -> tuple[dict, dict]:
    with _CACHE_LOCK:
        groups = cache.get("by_group", {})
        cands = []
        for gid, m in groups.items():
            if qno_a in m and qno_b in m:
                cands.append((gid, m[qno_a], m[qno_b]))
        if not cands:
            raise LookupError(f"Không tìm thấy bài có đủ question_no={qno_a} và {qno_b}.")
        recent_group_ids = set(_RECENT_BY_USER[hist_key])
        filtered = [c for c in cands if c[0] not in recent_group_ids]
        if not filtered:
            _RECENT_BY_USER[hist_key].clear()
            filtered = cands
        gid, rec_a, rec_b = random.choice(filtered)
        _RECENT_BY_USER[hist_key].append(gid)
        return rec_a, rec_b

def choose_group_by_qnos(cache: dict, qnos: Iterable[int], hist_key: tuple) -> list[dict]:
    """
    Chọn một 'bài' (parent_id) có ĐỦ tất cả qnos trong nhóm; tránh lặp theo group.
    Trả về danh sách rec (cùng thứ tự qnos đã sort).
    """
    target = set(int(x) for x in qnos)
    with _CACHE_LOCK:
        groups = cache.get("by_group", {})
        cands = []
        for gid, m in groups.items():
            if target.issubset(m.keys()):
                ordered = [m[q] for q in sorted(target)]
                cands.append((gid, ordered))
        if not cands:
            raise LookupError(f"Không tìm thấy bài có đủ các câu: {sorted(target)}.")

        recent_group_ids = set(_RECENT_BY_USER[hist_key])
        filtered = [c for c in cands if c[0] not in recent_group_ids]
        if not filtered:
            _RECENT_BY_USER[hist_key].clear()
            filtered = cands

        gid, recs = random.choice(filtered)
        _RECENT_BY_USER[hist_key].append(gid)
        return recs
