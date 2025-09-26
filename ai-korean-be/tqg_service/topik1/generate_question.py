# topik_service.py
import json, os, time, random, hashlib, threading, re
from pathlib import Path
from collections import deque, defaultdict
from typing import Dict, Optional, List, Tuple

# ====== CẤU HÌNH DỮ LIỆU ======
TOPIKI14_LISTEN_DIR = Path(os.getenv("TOPIKI14_LISTEN_DIR", "data/topik1/listen/1-4/OUT")).resolve()
TOPIKI56_LISTEN_DIR = Path(os.getenv("TOPIKI56_LISTEN_DIR", "data/topik1/listen/5-6/OUT")).resolve()


# ---- ROUTING RULES (data-driven) ----
# Mỗi mục gồm các "ranges" quyết định thư mục, yêu cầu question_no, và fallback score.
RULES = {
    ("topik1", "listening"): {
        "ranges": [
            # Câu 1–4: đọc từ TOPIKI14_LISTEN_DIR, ưu tiên question_no (nếu thiếu → fallback score 4/4/3/3)
            {"cau_lo": 1, "cau_hi": 4, "dir": TOPIKI14_LISTEN_DIR, "qno_required": False,
             "fallback_score_by_cau": {1: 4, 2: 4, 3: 3, 4: 3}},
            # Câu 5–6: đọc từ TOPIKI56_LISTEN_DIR, BẮT BUỘC có question_no
            {"cau_lo": 5, "cau_hi": 6, "dir": TOPIKI56_LISTEN_DIR, "qno_required": True,
             "fallback_score_by_cau": {5: 4, 6: 3}},
        ]
    },
    # Sau này chỉ cần thêm:
    # ("topik1", "reading"): { "ranges": [ { ... } ] },
    # ("topik2", "listening"): { "ranges": [ { ... } ] },
}

# điểm mục tiêu theo câu (fallback khi item không có score)
TARGET_SCORE_BY_CAU = {1: 4, 2: 4, 3: 3, 4: 3, 5: 4, 6: 3}

# ====== TITLES MAPPING (theo khoảng câu) ======
_TITLES_RANGES: List[Tuple[Tuple[int,int], str]] = [
    ((1, 4),  "※[1~4] 다음을 듣고 <보기>와 같이 물음에 맞는 대답을 고르십시오."),
    ((5, 6),  "※[5~6] 다음을 듣고 <보기>와 같이 이어지는 말을 고르십시오."),
    ((7, 10), "※[7~10] 여기는 어디입니까? <보기>와 같이 알맞은 것을 고르십시오."),
    ((11,14), "※[11~14] 다음은 무엇에 대해 말하고 있습니까? <보기>와 같이 알맞은 것을 고르십시오."),
    ((15,16), "※[15~16] 다음을 듣고 가장 알맞은 그림을 고르십시오. (각 4점)"),
    ((17,21), "※[17~21] 다음을 듣고 <보기>와 같이 대화 내용과 같은 것을 고르십시오. (각 3점)"),
    ((22,24), "※[22~24] 다음을 듣고 여자의 중심 생각을 고르십시오. (각 3점)"),
    ((25,26), "※[25~26] 다음을 듣고 물음에 답하십시오."),
    ((27,28), "※[27~28] 다음을 듣고 물음에 답하십시오."),
    ((29,30), "※[29~30] 다음을 듣고 물음에 답하십시오."),
]

def _title_for_cau(cau: int, fallback: str | None = None) -> str:
    for (lo, hi), title in _TITLES_RANGES:
        if lo <= cau <= hi:
            return title
    return fallback or f"Câu {cau}"

# ====== CACHE (lưu theo question_no; fallback score) ======
_CACHE_LOCK = threading.RLock()

# Cache theo thư mục:
#  - by_qno: {1:[...],...,30:[...]} -> ƯU TIÊN nếu item có 'question_no'
#  - by_score: {3:[...],4:[...]}    -> fallback
_DIR_CACHE: Dict[Path, dict] = defaultdict(lambda: {
    "loaded_at": 0.0,
    "files": [],
    "mtimes": {},
    "by_qno": {i: [] for i in range(1, 31)},
    "by_score": {3: [], 4: []},
})

# Lịch sử tránh trùng theo (user_key, danh_muc, key, dir)
_RECENT_BY_USER: Dict[tuple, deque] = defaultdict(lambda: deque(maxlen=200))

# ====== TIỆN ÍCH ======
def _stable_id(obj: dict) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

def _load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

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

def _get_qno_strict(obj: dict) -> Optional[int]:
    """
    ƯU TIÊN khóa 'question_no' đúng như yêu cầu.
    Trả về int 1..30 nếu có và hợp lệ; ngược lại trả về None.
    """
    v = obj.get("question_no")
    if v is None:
        return None
    try:
        iv = int(v)
        if 1 <= iv <= 30:
            return iv
    except Exception:
        pass
    return None

def _get_score_optional(obj: dict) -> Optional[int]:
    v = obj.get("score")
    if v is None:
        return None
    try:
        iv = int(v)
        return iv if iv in (3, 4) else None
    except Exception:
        return None

def _refresh_cache(dir_path: Path):
    with _CACHE_LOCK:
        by_qno = {i: [] for i in range(1, 31)}
        by_score = {3: [], 4: []}

        files = list(dir_path.glob("*.jsonl"))
        mtimes = {p: p.stat().st_mtime for p in files}

        for p in files:
            for obj in _load_jsonl(p):
                qid = obj.get("id") or _stable_id(obj)
                rec = {"id": qid, "data": obj}

                # 1) Nạp theo question_no nếu có
                qno = _get_qno_strict(obj)
                if qno is not None:
                    by_qno[qno].append(rec)

                # 2) Đồng thời nạp theo score (để fallback khi thiếu qno)
                sc = _get_score_optional(obj)
                if sc is not None:
                    by_score[sc].append(rec)

        # shuffle để rút đều
        for i in range(1, 31):
            random.shuffle(by_qno[i])
        random.shuffle(by_score[3])
        random.shuffle(by_score[4])

        _DIR_CACHE[dir_path].update({
            "by_qno": by_qno,
            "by_score": by_score,
            "files": files,
            "mtimes": mtimes,
            "loaded_at": time.time(),
        })

def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()

def _pick_rule(level: str, danh_muc: str, cau: int):
    key = (_norm(level), _norm(danh_muc))
    cfg = RULES.get(key)
    if not cfg:
        return None, None
    for r in cfg.get("ranges", []):
        if r["cau_lo"] <= cau <= r["cau_hi"]:
            return cfg, r
    return cfg, None

def warm_topik14_listen_cache():
    if TOPIKI14_LISTEN_DIR.exists():
        _refresh_cache(TOPIKI14_LISTEN_DIR)

def warm_topik56_listen_cache():
    if TOPIKI56_LISTEN_DIR.exists():
        _refresh_cache(TOPIKI56_LISTEN_DIR)

def _choose_nonrepeating(pool: List[dict], hist_key: tuple) -> dict:
    with _CACHE_LOCK:
        if not pool:
            raise ValueError("Pool rỗng.")
        recent_ids = set(_RECENT_BY_USER[hist_key])
        candidates = [q for q in pool if q["id"] not in recent_ids]
        if not candidates:
            _RECENT_BY_USER[hist_key].clear()
            candidates = pool
        choice = random.choice(candidates)
        _RECENT_BY_USER[hist_key].append(choice["id"])
        return choice

# ====== Chuẩn hóa fields cho response ======
_SPEAKER_RE = re.compile(r"^(남자|여자|학생|선생님|교사|해설자|남|여)[:：]\s*(.+)$")

def _extract_question_text(obj: dict) -> str:
    return (
        obj.get("question")
        or obj.get("question_ko")
        or obj.get("prompt")
        or obj.get("prompt_ko")
        or ""
    )

def _extract_dialog(obj: dict) -> List[dict]:
    if isinstance(obj.get("dialogue"), list) and obj["dialogue"]:
        out = []
        for turn in obj["dialogue"]:
            if isinstance(turn, dict):
                spk = turn.get("speaker") or turn.get("role") or ""
                txt = turn.get("text") or turn.get("utterance") or ""
                if txt:
                    out.append({"speaker": spk, "text": txt})
        if out:
            return out

    passage = obj.get("passage_ko")
    if passage is None:
        return []

    out: List[dict] = []
    if isinstance(passage, str):
        lines = [ln.strip() for ln in re.split(r"[\r\n]+", passage) if ln.strip()]
        for ln in lines:
            m = _SPEAKER_RE.match(ln)
            if m:
                spk, txt = m.group(1), m.group(2).strip()
                out.append({"speaker": spk, "text": txt})
            else:
                out.append({"speaker": "지문", "text": ln})
        return out

    if isinstance(passage, list):
        for item in passage:
            if isinstance(item, dict):
                spk = item.get("speaker") or item.get("role") or "지문"
                txt = item.get("text") or item.get("utterance") or ""
                if txt:
                    out.append({"speaker": spk, "text": txt})
            elif isinstance(item, str):
                m = _SPEAKER_RE.match(item.strip())
                if m:
                    spk, txt = m.group(1), m.group(2).strip()
                    out.append({"speaker": spk, "text": txt})
                else:
                    out.append({"speaker": "지문", "text": item.strip()})
        return out

    return []

def _normalize_choices(obj: dict) -> Dict[str, str]:
    src = obj.get("choices")
    if src is None:
        src = obj.get("options")
    if src is None:
        return {}

    if isinstance(src, dict):
        norm = {}
        for k, v in src.items():
            if not isinstance(v, str):
                continue
            key = str(k).strip().upper()
            if len(key) == 1 and "A" <= key <= "Z":
                norm[key] = v
        if not norm and all(str(k).isdigit() for k in src.keys()):
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for i, k in enumerate(sorted(src.keys(), key=lambda x: int(x))):
                if i < len(letters) and isinstance(src[k], str):
                    norm[letters[i]] = src[k]
        return norm

    if isinstance(src, list):
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return {letters[i]: s for i, s in enumerate(src) if isinstance(s, str)}
    return {}

def _normalize_answer(obj: dict, choices: Dict[str, str]) -> str:
    ans = obj.get("answer")
    if ans is None:
        return "A" if choices else ""

    if isinstance(ans, str):
        up = ans.strip().upper()
        if up in choices:
            return up
        for k, v in choices.items():
            if v.strip() == ans.strip():
                return k
        if up.isdigit():
            idx = int(up)
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            if 1 <= idx <= len(choices):
                return letters[idx - 1]
            if 0 <= idx < len(choices):
                return letters[idx]
        return "A" if choices else ""

    if isinstance(ans, int):
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if 1 <= ans <= len(choices):
            return letters[ans - 1]
        if 0 <= ans < len(choices):
            return letters[ans]
        return "A" if choices else ""

    return "A" if choices else ""

# ====== API CHÍNH ======
class DataNotFoundError(Exception): ...
class BadRequestError(Exception): ...

def _ensure_ready(dir_path: Path):
    if not dir_path.exists():
        raise DataNotFoundError(f"Không thấy thư mục: {dir_path}")
    if _needs_refresh(dir_path):
        _refresh_cache(dir_path)

def generate_topik_question(
    level: str,
    danh_muc: str,
    cau: int,
    *,
    user_key: Optional[str] = "global",
) -> dict:
    """
    Router theo RULES:
      - Tìm cấu hình theo (level, danh_muc),
      - Chọn range theo 'cau',
      - Với mỗi range: ưu tiên rút theo question_no; nếu qno_required=True mà thiếu -> lỗi,
        nếu qno_required=False và pool qno rỗng -> fallback theo score (nếu có).
    """
    cau = int(cau)
    if not (1 <= cau <= 30):
        raise BadRequestError("Câu phải nằm trong khoảng 1..30.")

    cfg, rule = _pick_rule(level, danh_muc, cau)
    if cfg is None:
        raise BadRequestError(f"Chưa hỗ trợ level='{level}', danh_muc='{danh_muc}'. Hãy thêm RULES cho tổ hợp này.")
    if rule is None:
        raise BadRequestError(f"Chưa định nghĩa range cho câu {cau} ở ({level}, {danh_muc}).")

    dir_path: Path = rule["dir"]
    qno_required: bool = bool(rule.get("qno_required", False))
    fb_scores: Dict[int, int] = rule.get("fallback_score_by_cau", {})

    _ensure_ready(dir_path)
    cache = _DIR_CACHE[dir_path]

    # Ưu tiên pool theo question_no
    pool_qno = cache["by_qno"].get(cau, [])

    if qno_required:
        if not pool_qno:
            raise DataNotFoundError(
                f"Dataset {dir_path} không có item nào với question_no={cau}. "
                "Hãy đảm bảo mỗi item trong range này đều có 'question_no' hợp lệ."
            )
        hist_key = (user_key or "global", _norm(danh_muc), f"qno:{cau}", str(dir_path))
        chosen = _choose_nonrepeating(pool_qno, hist_key)
    else:
        if pool_qno:
            hist_key = (user_key or "global", _norm(danh_muc), f"qno:{cau}", str(dir_path))
            chosen = _choose_nonrepeating(pool_qno, hist_key)
        else:
            # Fallback theo score (nếu được định nghĩa)
            score = fb_scores.get(cau)
            if score is None:
                raise DataNotFoundError(
                    f"Thiếu question_no={cau} và không có fallback score cho range này trong {dir_path}."
                )
            pool_score = cache["by_score"].get(score, [])
            if not pool_score:
                raise DataNotFoundError(
                    f"Không có item fallback score={score} cho câu {cau} trong {dir_path}."
                )
            hist_key = (user_key or "global", _norm(danh_muc), f"score:{score}", str(dir_path))
            chosen = _choose_nonrepeating(pool_score, hist_key)

    data = chosen["data"]

    # Build output
    title = _title_for_cau(cau, fallback=data.get("title"))
    qtext = _extract_question_text(data)
    dialog = _extract_dialog(data)
    choices = _normalize_choices(data)
    answer = _normalize_answer(data, choices)

    # Điểm: ưu tiên field 'score' của item; nếu thiếu thì theo TARGET_SCORE_BY_CAU / fb_scores
    score = data.get("score")
    try:
        score = int(score) if score is not None else None
    except Exception:
        score = None
    if score is None:
        score = TARGET_SCORE_BY_CAU.get(cau) or fb_scores.get(cau) or 4

    return {
        "title": title,
        "type":  data.get("type", danh_muc),
        "question": qtext or "",
        "dialog":  dialog,
        "choices": choices,
        "answer":  answer,
        "score":   int(score),
    }
