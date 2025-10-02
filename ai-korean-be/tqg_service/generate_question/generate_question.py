# generate_question/generate_question.py
from __future__ import annotations
from typing import Dict, Optional, Tuple

from pathlib import Path

from .config import (
    RULES, LEVEL_CAU_LIMITS, DEFAULT_SCORE_BY_LEVEL, get_group_for
)
from .picker import choose_nonrepeating, choose_pair_by_group, choose_group_by_qnos
from .builder import build_single_from_rec, build_pair_record, build_group_record
from .utils import norm
from .cache import ensure_ready, refresh_cache, _DIR_CACHE

class DataNotFoundError(Exception): ...
class BadRequestError(Exception): ...


def _max_cau_for_level(level: str) -> Optional[int]:
    return LEVEL_CAU_LIMITS.get((level or "").strip().lower())


def _pick_rule(level: str, danh_muc: str, cau: int):
    key = (norm(level), norm(danh_muc))
    cfg = RULES.get(key)
    if not cfg:
        return None, None
    for r in cfg.get("ranges", []):
        if r["cau_lo"] <= cau <= r["cau_hi"]:
            return cfg, r
    return cfg, None


def warm_all_caches() -> None:
    # iterate over all dirs in RULES and refresh
    seen = set()
    for (_, _), cfg in RULES.items():
        for r in cfg.get("ranges", []):
            d: Path = r["dir"]
            if d not in seen and d.exists():
                refresh_cache(d)
                seen.add(d)


def generate_topik_question(level: str, danh_muc: str, cau: int, *, user_key: Optional[str] = "global") -> dict:
    cau = int(cau)
    lvl = norm(level)
    max_cau = _max_cau_for_level(lvl)
    if max_cau is None:
        raise BadRequestError(f"Level '{level}' chưa được hỗ trợ. Hãy thêm vào LEVEL_CAU_LIMITS hoặc RULES.")
    if not (1 <= cau <= max_cau):
        raise BadRequestError(f"Câu phải nằm trong khoảng 1..{max_cau} cho level='{level}'.")

    cfg, rule = _pick_rule(level, danh_muc, cau)
    if cfg is None:
        raise BadRequestError(f"Chưa hỗ trợ level='{level}', danh_muc='{danh_muc}'. Hãy thêm RULES cho tổ hợp này.")
    if rule is None:
        raise BadRequestError(f"Chưa định nghĩa range cho câu {cau} ở ({level}, {danh_muc}).")

    dir_path: Path = rule["dir"]
    ensure_ready(dir_path)
    cache = _DIR_CACHE[dir_path]

    # === Nhóm nếu thuộc cặp ===
    group = get_group_for(level, danh_muc, cau)  # tuple như (48,49,50) hoặc (63,64) ...
    if group and len(group) >= 2:
        # Chỉ chọn group nếu câu hiện tại thuộc đúng range rule hiện tại (để không chệch dir)
        if all(rule["cau_lo"] <= q <= rule["cau_hi"] for q in group):
            hist_key = (user_key or "global", norm(danh_muc), f"group:{'-'.join(map(str,group))}", str(dir_path))
            recs = choose_group_by_qnos(cache, group, hist_key)
            return build_group_record(level, danh_muc, rule, recs, default_score_by_level=DEFAULT_SCORE_BY_LEVEL)

    # === Đơn lẻ: ưu tiên qno, fallback score ===
    qno_required: bool = bool(rule.get("qno_required", False))
    fb_scores: Dict[int, int] = rule.get("fallback_score_by_cau", {})
    pool_qno = cache["by_qno"].get(cau, [])

    if qno_required:
        if not pool_qno:
            raise DataNotFoundError(
                f"Dataset {dir_path} không có item nào với question_no={cau}. "
                "Hãy đảm bảo mỗi item trong range này đều có 'question_no' hợp lệ."
            )
        hist_key = (user_key or "global", norm(danh_muc), f"qno:{cau}", str(dir_path))
        chosen = choose_nonrepeating(pool_qno, hist_key)
    else:
        if pool_qno:
            hist_key = (user_key or "global", norm(danh_muc), f"qno:{cau}", str(dir_path))
            chosen = choose_nonrepeating(pool_qno, hist_key)
        else:
            score_needed = fb_scores.get(cau)
            if score_needed is None:
                raise DataNotFoundError(
                    f"Thiếu question_no={cau} và không có fallback score cho range này trong {dir_path}."
                )
            pool_score = cache["by_score"].get(score_needed, [])
            if not pool_score:
                raise DataNotFoundError(
                    f"Không có item fallback score={score_needed} cho câu {cau} trong {dir_path}."
                )
            hist_key = (user_key or "global", norm(danh_muc), f"score:{score_needed}", str(dir_path))
            chosen = choose_nonrepeating(pool_score, hist_key)

    data = chosen["data"]
    return build_single_from_rec(level, danh_muc, rule, data, default_score_by_level=DEFAULT_SCORE_BY_LEVEL, fallback_cau=cau)