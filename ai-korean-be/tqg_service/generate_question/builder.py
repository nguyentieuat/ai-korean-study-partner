# generate_question/builder.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

from .utils import norm, extract_question_text, extract_passage, extract_dialog, normalize_choices, normalize_answer

# điểm mặc định theo level — truyền từ config ở chỗ gọi

def is_doc_like(d: dict) -> bool:
    section = (d.get("section") or "").strip().lower()
    t       = norm(d.get("type"))
    return section == "doc" or t in ("reading", "doc")


def pick_group_title(*titles: Optional[str], lo: Optional[int] = None, hi: Optional[int] = None) -> str:
    for t in titles:
        if isinstance(t, str) and t.strip():
            return t.strip()
    if lo is not None and hi is not None and lo and hi:
        return f"※ [{lo}~{hi}] 문항"
    return "문항"


def build_single_from_rec(level: str, danh_muc: str, rule: dict, data: dict, *, default_score_by_level: Dict[str, int], fallback_cau: int) -> dict:
    title   = data.get("title") or f"Câu {int(data.get('question_no') or fallback_cau)}"
    qtext   = extract_question_text(data)
    choices = normalize_choices(data)
    answer  = normalize_answer(data, choices)

    raw_score = data.get("score")
    try:
        item_score = int(raw_score) if raw_score is not None else None
    except Exception:
        item_score = None

    fb_scores     = rule.get("fallback_score_by_cau", {})
    default_score = default_score_by_level.get(norm(level), 3)
    score = item_score if item_score is not None else fb_scores.get(int(data.get("question_no") or fallback_cau), default_score)

    section = (data.get("section") or "").strip().lower()
    is_doc  = (section == "doc") or (norm(danh_muc) == "reading")

    out = {
        "title":       title,
        "type":        data.get("type", danh_muc),
        "question":    qtext or "",
        "choices":     choices,
        "answer":      answer,
        "score":       int(score if score is not None else default_score),
        "question_no": int(data.get("question_no") or 0),
    }
    if is_doc:
        out["passage"] = extract_passage(data)
    else:
        out["dialog"]  = extract_dialog(data)
    return out


def build_pair_record(level: str, danh_muc: str, rule: dict, rec_a: dict, rec_b: dict, *, default_score_by_level: Dict[str, int]) -> dict:
    data_a, data_b = rec_a["data"], rec_b["data"]
    item_a = build_single_from_rec(level, danh_muc, rule, data_a, default_score_by_level=default_score_by_level, fallback_cau=25)
    item_b = build_single_from_rec(level, danh_muc, rule, data_b, default_score_by_level=default_score_by_level, fallback_cau=26)

    doc_like = is_doc_like(data_a) or is_doc_like(data_b)
    if doc_like:
        shared_key = "passage"
        shared_val = extract_passage(data_a) or extract_passage(data_b) or ""
    else:
        shared_key = "dialog"
        shared_val = extract_dialog(data_a) or extract_dialog(data_b) or []

    qa = item_a.get("question_no") or 0
    qb = item_b.get("question_no") or 0
    lo, hi = (qa, qb) if qa <= qb else (qb, qa)

    group_title = pick_group_title(data_a.get("title"), data_b.get("title"), lo=lo, hi=hi)

    strip_keys = {"dialog", "passage"}
    ia = {k: v for k, v in item_a.items() if k not in strip_keys}
    ib = {k: v for k, v in item_b.items() if k not in strip_keys}

    return {
        "title":        group_title,
        "type":         data_a.get("type", danh_muc),
        "items":        [ia, ib],
        "question_no":  [ia["question_no"], ib["question_no"]],
        "score_total":  (ia.get("score", 0) + ib.get("score", 0)),
        shared_key:     shared_val,
    }

def build_group_record(level: str, danh_muc: str, rule: dict, recs: list[dict], *, default_score_by_level: Dict[str, int]) -> dict:
    """
    recs: list các rec dạng {"id":..., "data": {...}} cho tất cả câu trong nhóm
    """
    # Build từng item (không kèm passage/dialog bên trong)
    items_raw = []
    titles = []
    doc_like_any = False
    passages = []
    dialogs = []
    for r in recs:
        d = r["data"]
        titles.append(d.get("title"))
        item = build_single_from_rec(level, danh_muc, rule, d, default_score_by_level=default_score_by_level,
                                     fallback_cau=int(d.get("question_no") or 0))
        # tách shared
        doc_like = is_doc_like(d)
        doc_like_any = doc_like_any or doc_like
        if doc_like:
            passages.append(extract_passage(d))
        else:
            dialogs.append(extract_dialog(d))
        for k in ("dialog", "passage"):
            item.pop(k, None)
        items_raw.append(item)

    # shared content
    if doc_like_any:
        shared_key = "passage"
        shared_val = next((p for p in passages if p), "")
    else:
        shared_key = "dialog"
        # hợp nhất hội thoại (nếu nhiều) — thường là 1
        shared_val = next((dg for dg in dialogs if dg), [])

    # tiêu đề nhóm
    qnos = [it.get("question_no") or 0 for it in items_raw]
    lo, hi = min(qnos), max(qnos)
    group_title = pick_group_title(*titles, lo=lo, hi=hi)

    return {
        "title": group_title,
        "type":  recs[0]["data"].get("type", danh_muc),
        "items": items_raw,
        "question_no": qnos,
        "score_total": sum(int(it.get("score", 0)) for it in items_raw),
        shared_key: shared_val,
    }
