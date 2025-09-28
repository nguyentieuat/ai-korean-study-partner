# generate_question/generate_question.py
import json, os, time, random, hashlib, threading, re
from pathlib import Path
from collections import deque, defaultdict
from typing import Dict, Optional, List, Tuple
from collections import defaultdict as _dd

# ====== CẤU HÌNH DỮ LIỆU ======
TOPIKI14_LISTEN_DIR   = Path(os.getenv("TOPIKI14_LISTEN_DIR",   "data/topik1/listen/1-4/OUT")).resolve()
TOPIKI56_LISTEN_DIR   = Path(os.getenv("TOPIKI56_LISTEN_DIR",   "data/topik1/listen/5-6/OUT")).resolve()
TOPIKI710_LISTEN_DIR  = Path(os.getenv("TOPIKI710_LISTEN_DIR",  "data/topik1/listen/7-10/OUT")).resolve()
TOPIKI1114_LISTEN_DIR = Path(os.getenv("TOPIKI1114_LISTEN_DIR", "data/topik1/listen/11-14/OUT")).resolve()
# TOPIKI1516_LISTEN_DIR = Path(os.getenv("TOPIKI1516_LISTEN_DIR", "data/topik1/listen/15-16/OUT")).resolve()
TOPIKI1721_LISTEN_DIR = Path(os.getenv("TOPIKI1721_LISTEN_DIR", "data/topik1/listen/17-21/OUT")).resolve()
TOPIKI2224_LISTEN_DIR = Path(os.getenv("TOPIKI2224_LISTEN_DIR", "data/topik1/listen/22-24/OUT")).resolve()
TOPIKI2526_LISTEN_DIR = Path(os.getenv("TOPIKI2526_LISTEN_DIR", "data/topik1/listen/25-26/OUT")).resolve()
TOPIKI2728_LISTEN_DIR = Path(os.getenv("TOPIKI2728_LISTEN_DIR", "data/topik1/listen/27-28/OUT")).resolve()
TOPIKI2930_LISTEN_DIR = Path(os.getenv("TOPIKI2930_LISTEN_DIR", "data/topik1/listen/29-30/OUT")).resolve()

TOPIKI3133_DOC_DIR    = Path(os.getenv("TOPIKI3133_DOC_DIR",    "data/topik1/reading/31-33/OUT")).resolve()
TOPIKI3439_DOC_DIR    = Path(os.getenv("TOPIKI3439_DOC_DIR",    "data/topik1/reading/34-39/OUT")).resolve()
# TOPIKI4042_DOC_DIR    = Path(os.getenv("TOPIKI4042_DOC_DIR",    "data/topik1/reading/40-42/OUT")).resolve()
TOPIKI4345_DOC_DIR    = Path(os.getenv("TOPIKI4345_DOC_DIR",    "data/topik1/reading/43-45/OUT")).resolve()
TOPIKI4648_DOC_DIR    = Path(os.getenv("TOPIKI4648_DOC_DIR",    "data/topik1/reading/46-48/OUT")).resolve()

# ---- ROUTING RULES (data-driven) ----
# Mỗi mục gồm các "ranges" quyết định thư mục, yêu cầu question_no, và fallback score.
RULES = {
    ("topik1", "listening"): {
        "ranges": [
            {"cau_lo": 1,  "cau_hi": 4,  "dir": TOPIKI14_LISTEN_DIR,   "qno_required": False,
             "fallback_score_by_cau": {1: 4, 2: 4, 3: 3, 4: 3}},
            {"cau_lo": 5,  "cau_hi": 6,  "dir": TOPIKI56_LISTEN_DIR,   "qno_required": True,
             "fallback_score_by_cau": {5: 4, 6: 3}},
            {"cau_lo": 7,  "cau_hi": 10, "dir": TOPIKI710_LISTEN_DIR,  "qno_required": False,
             "fallback_score_by_cau": {7: 3, 8: 3, 9: 3, 10: 4}},
            {"cau_lo": 11, "cau_hi": 14, "dir": TOPIKI1114_LISTEN_DIR, "qno_required": False,
             "fallback_score_by_cau": {11: 3, 12: 3, 13: 4, 14: 3}},
            # {"cau_lo": 15, "cau_hi": 16, "dir": TOPIKI1516_LISTEN_DIR, "qno_required": False,
            #  "fallback_score_by_cau": {15: 4, 16: 4}},
            {"cau_lo": 17, "cau_hi": 21, "dir": TOPIKI1721_LISTEN_DIR, "qno_required": False,
             "fallback_score_by_cau": {17: 3, 18: 3, 19: 3, 20: 3, 21: 3}},
            {"cau_lo": 22, "cau_hi": 24, "dir": TOPIKI2224_LISTEN_DIR, "qno_required": False,
             "fallback_score_by_cau": {22: 3, 23: 3, 24: 3}},
            {"cau_lo": 25, "cau_hi": 26, "dir": TOPIKI2526_LISTEN_DIR, "qno_required": False,
             "fallback_score_by_cau": {25: 3, 26: 4}},
            {"cau_lo": 27, "cau_hi": 28, "dir": TOPIKI2728_LISTEN_DIR, "qno_required": False,
             "fallback_score_by_cau": {27: 3, 28: 4}},
            {"cau_lo": 29, "cau_hi": 30, "dir": TOPIKI2930_LISTEN_DIR, "qno_required": False,
             "fallback_score_by_cau": {29: 3, 30: 4}},
        ]
    },
    ("topik1", "reading"): {
        "ranges": [
            {"cau_lo": 31, "cau_hi": 33, "dir": TOPIKI3133_DOC_DIR, "qno_required": False,
             "fallback_score_by_cau": {31: 2, 32: 2, 33: 2}},
            {"cau_lo": 34, "cau_hi": 39, "dir": TOPIKI3439_DOC_DIR, "qno_required": False,
             "fallback_score_by_cau": {34: 2, 35: 2, 36: 2, 37: 3, 38: 3, 39: 2}},
            # {"cau_lo": 40, "cau_hi": 42, "dir": TOPIKI4042_DOC_DIR, "qno_required": False,
            # "fallback_score_by_cau": {40: 2, 41: 2, 42: 2}},
            {"cau_lo": 43, "cau_hi": 45, "dir": TOPIKI4345_DOC_DIR, "qno_required": False,
            "fallback_score_by_cau": {43: 3, 44: 2, 45: 3}},
            {"cau_lo": 46, "cau_hi": 48, "dir": TOPIKI4648_DOC_DIR, "qno_required": False,
            "fallback_score_by_cau": {46: 3, 47: 3, 48: 2}},
        ]
    },
    # ("topik2", "listening"): { "ranges": [ { ... } ] },
}

# ====== GIỚI HẠN CÂU THEO LEVEL ======
LEVEL_CAU_LIMITS = {
    "topik1": 70,
    "topik2": 50,
}
def _max_cau_for_level(level: str) -> Optional[int]:
    return LEVEL_CAU_LIMITS.get((level or "").strip().lower())

# ====== ĐIỂM MẶC ĐỊNH THEO LEVEL ======
DEFAULT_SCORE_BY_LEVEL = {
    "topik1": 3,
    "topik2": 2,
}
def _default_score_for_level(level: str) -> int:
    return DEFAULT_SCORE_BY_LEVEL.get(_norm(level), 3)

# ====== TITLES MAPPING (theo khoảng câu) ======
_TITLES_RANGES: List[Tuple[Tuple[int,int], str]] = [
    ((1, 4),  "※[1~4] 다음을 듣고 물음에 맞는 대답을 고르십시오."),
    ((5, 6),  "※[5~6] 다음을 듣고 이어지는 말을 고르십시오."),
    ((7, 10), "※[7~10] 여기는 어디입니까? 알맞은 것을 고르십시오."),
    ((11,14), "※[11~14] 다음은 무엇에 대해 말하고 있습니까? 알맞은 것을 고르십시오."),
    ((15,16), "※[15~16] 다음을 듣고 가장 알맞은 그림을 고르십시오."),
    ((17,21), "※[17~21] 다음을 듣고 대화 내용과 같은 것을 고르십시오."),
    ((22,24), "※[22~24] 다음을 듣고 여자의 중심 생각을 고르십시오."),
    ((25,26), "※[25~26] 다음을 듣고 물음에 답하십시오."),
    ((27,28), "※[27~28] 다음을 듣고 물음에 답하십시오."),
    ((29,30), "※[29~30] 다음을 듣고 물음에 답하십시오."),
    ((31,33), "※[31~33] 무엇에 대한 내용입니까? 알맞은 것을 고르십시오."),
    ((34,39), "※[34~39] ( )에 들어갈 말로 가장 알맞은 것을 고르십시오."),
]

def _title_for_cau(cau: int, fallback: str | None = None) -> str:
    for (lo, hi), title in _TITLES_RANGES:
        if lo <= cau <= hi:
            return title
    return fallback or f"Câu {cau}"

# ====== CACHE (lưu theo question_no; fallback score; group) ======
_CACHE_LOCK = threading.RLock()

# Cache theo thư mục:
#  - by_qno:   dynamic {qno: [records]}
#  - by_score: {2:[...],3:[...],4:[...]}      -> fallback
#  - by_group: {parent_id: {qno: rec}}
_DIR_CACHE: Dict[Path, dict] = defaultdict(lambda: {
    "loaded_at": 0.0,
    "files": [],
    "mtimes": {},
    "by_qno": _dd(list),                # dynamic
    "by_score": {2: [], 3: [], 4: []},  # gồm score=2 cho Doc
    "by_group": defaultdict(dict),
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
    Trả về int 1..70 nếu có và hợp lệ; ngược lại trả về None.
    """
    v = obj.get("question_no")
    if v is None:
        return None
    try:
        iv = int(v)
        if 1 <= iv <= 70:
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
        return iv if iv in (2, 3, 4) else None
    except Exception:
        return None

# --- helper: làm sạch " x" đánh dấu ở cuối option (nếu dữ liệu có) ---
def _clean_option_text(s: str) -> str:
    if not isinstance(s, str):
        return s
    return re.sub(r'\s*x\s*$', '', s.strip(), flags=re.IGNORECASE)

def _expand_multi_question(obj: dict):
    """
    Nếu obj có mảng 'questions', tách thành nhiều record con và gắn 'parent_id' để nhóm cặp 25&26, 27&28, 29&30.
    Nếu không có 'questions', yield chính obj.
    """
    qs = obj.get("questions")
    if not isinstance(qs, list) or not qs:
        yield obj
        return

    parent_id = obj.get("id") or _stable_id(obj)  # nhận diện bài gốc
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
            opts = {k: _clean_option_text(v) for k, v in opts.items() if isinstance(v, str)}
        elif isinstance(opts, list):
            opts = [_clean_option_text(v) for v in opts if isinstance(v, str)]

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

def _refresh_cache(dir_path: Path):
    with _CACHE_LOCK:
        by_qno   = _dd(list)
        by_score = {2: [], 3: [], 4: []}
        by_group = defaultdict(dict)

        files  = list(dir_path.glob("*.jsonl"))
        mtimes = {p: p.stat().st_mtime for p in files}

        for p in files:
            for obj in _load_jsonl(p):
                # Expand nếu là multi-question
                for sub in _expand_multi_question(obj):
                    qid = sub.get("id") or _stable_id(sub)
                    rec = {"id": qid, "data": sub}

                    qno = _get_qno_strict(sub)
                    if qno is not None:
                        by_qno[qno].append(rec)

                    sc = _get_score_optional(sub)
                    if sc is not None and sc in by_score:
                        by_score[sc].append(rec)

                    pid = sub.get("parent_id")
                    if pid and qno is not None:
                        by_group[pid][qno] = rec

        # shuffle đều các bucket hiện có
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

def warm_topik710_listen_cache():
    if TOPIKI710_LISTEN_DIR.exists():
        _refresh_cache(TOPIKI710_LISTEN_DIR)

def warm_topik1114_listen_cache():
    if TOPIKI1114_LISTEN_DIR.exists():
        _refresh_cache(TOPIKI1114_LISTEN_DIR)

# def warm_topik1516_listen_cache():
#     if TOPIKI1516_LISTEN_DIR.exists():
#         _refresh_cache(TOPIKI1516_LISTEN_DIR)

def warm_topik1721_listen_cache():
    if TOPIKI1721_LISTEN_DIR.exists():
        _refresh_cache(TOPIKI1721_LISTEN_DIR)

def warm_topik2224_listen_cache():
    if TOPIKI2224_LISTEN_DIR.exists():
        _refresh_cache(TOPIKI2224_LISTEN_DIR)

def warm_topik2526_listen_cache():
    if TOPIKI2526_LISTEN_DIR.exists():
        _refresh_cache(TOPIKI2526_LISTEN_DIR)

def warm_topik2728_listen_cache():
    if TOPIKI2728_LISTEN_DIR.exists():
        _refresh_cache(TOPIKI2728_LISTEN_DIR)

def warm_topik2930_listen_cache():
    if TOPIKI2930_LISTEN_DIR.exists():
        _refresh_cache(TOPIKI2930_LISTEN_DIR)

def warm_topik3133_listen_cache():
    if TOPIKI3133_DOC_DIR.exists():
        _refresh_cache(TOPIKI3133_DOC_DIR)

def warm_topik3439_listen_cache():
    if TOPIKI3439_DOC_DIR.exists():
        _refresh_cache(TOPIKI3439_DOC_DIR)

# def warm_topik4042_listen_cache():
#     if TOPIKI4042_DOC_DIR.exists():
#         _refresh_cache(TOPIKI4042_DOC_DIR)

def warm_topik4345_listen_cache():
    if TOPIKI4345_DOC_DIR.exists():
        _refresh_cache(TOPIKI4345_DOC_DIR)

def warm_topik4648_listen_cache():
    if TOPIKI4648_DOC_DIR.exists():
        _refresh_cache(TOPIKI4648_DOC_DIR)

def warm_all_caches():
    warm_topik14_listen_cache()
    warm_topik56_listen_cache()
    warm_topik710_listen_cache()
    warm_topik1114_listen_cache()
    # warm_topik1516_listen_cache()
    warm_topik1721_listen_cache()
    warm_topik2224_listen_cache()
    warm_topik2526_listen_cache()
    warm_topik2728_listen_cache()
    warm_topik2930_listen_cache()
    warm_topik3133_listen_cache()
    warm_topik3439_listen_cache()
    # warm_topik4042_listen_cache()
    warm_topik4345_listen_cache()
    warm_topik4648_listen_cache()

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

def _choose_pair_by_group(cache: dict, qno_a: int, qno_b: int, hist_key: tuple) -> Tuple[dict, dict]:
    """Chọn một 'bài' (parent_id) có đủ qno_a và qno_b; tránh lặp theo group."""
    with _CACHE_LOCK:
        groups = cache.get("by_group", {})
        cands = []
        for gid, m in groups.items():
            if qno_a in m and qno_b in m:
                cands.append((gid, m[qno_a], m[qno_b]))
        if not cands:
            raise DataNotFoundError(f"Không tìm thấy bài có đủ question_no={qno_a} và {qno_b}.")

        recent_group_ids = set(_RECENT_BY_USER[hist_key])
        filtered = [c for c in cands if c[0] not in recent_group_ids]
        if not filtered:
            _RECENT_BY_USER[hist_key].clear()
            filtered = cands

        gid, rec_a, rec_b = random.choice(filtered)
        _RECENT_BY_USER[hist_key].append(gid)
        return rec_a, rec_b

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
    """
    Chuẩn hoá hội thoại về dạng:
      [ {"speaker": "...", "text": "..."}, ... ]
    Hỗ trợ các trường hợp:
      - dialogue: list[dict | str]
      - dialogue: dict (1 lượt hoặc bọc field 'turns')
      - dialogue: str
      - passage_ko: str | list[str|dict]
    """
    dlg = obj.get("dialogue")

    # ---- 1) Ưu tiên 'dialogue' nếu có ----
    # dialogue là list
    if isinstance(dlg, list) and dlg:
        out: List[dict] = []
        for turn in dlg:
            if isinstance(turn, dict):
                spk = turn.get("speaker") or turn.get("role") or "지문"
                txt = turn.get("text") or turn.get("utterance") or ""
                if txt:
                    out.append({"speaker": spk, "text": txt})
            elif isinstance(turn, str):
                t = turn.strip()
                if t:
                    out.append({"speaker": "지문", "text": t})
        if out:
            return out

    # dialogue là dict (trường hợp dữ liệu 1 lượt, hoặc bọc 'turns')
    if isinstance(dlg, dict):
        # Trường hợp có 'turns' bên trong
        inner = dlg.get("turns")
        if isinstance(inner, list) and inner:
            out: List[dict] = []
            for turn in inner:
                if isinstance(turn, dict):
                    spk = turn.get("speaker") or turn.get("role") or "지문"
                    txt = turn.get("text") or turn.get("utterance") or ""
                    if txt:
                        out.append({"speaker": spk, "text": txt})
                elif isinstance(turn, str):
                    t = turn.strip()
                    if t:
                        out.append({"speaker": "지문", "text": t})
            if out:
                return out
        # Trường hợp 1 lượt đơn
        spk = dlg.get("speaker") or dlg.get("role") or "지문"
        txt = dlg.get("text") or dlg.get("utterance") or ""
        if isinstance(txt, str) and txt.strip():
            return [{"speaker": spk, "text": txt.strip()}]

    # dialogue là str
    if isinstance(dlg, str):
        t = dlg.strip()
        if t:
            return [{"speaker": "지문", "text": t}]

    # ---- 2) Fallback: 'passage_ko' ----
    passage = obj.get("passage_ko")
    if passage is None:
        return []

    _speaker_re_local = re.compile(r"^(남자|여자|학생|선생님|교사|해설자|남|여)[:：]\s*(.+)$")

    out: List[dict] = []
    if isinstance(passage, str):
        lines = [ln.strip() for ln in re.split(r"[\r\n]+", passage) if ln.strip()]
        for ln in lines:
            m = _speaker_re_local.match(ln)
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
                m = _speaker_re_local.match(item.strip())
                if m:
                    spk, txt = m.group(1), m.group(2).strip()
                    out.append({"speaker": spk, "text": txt})
                else:
                    out.append({"speaker": "지문", "text": item.strip()})
        return out

    return []

def _extract_passage(obj: dict) -> str:
    # Ưu tiên 'passage' (tiếng Hàn), fallback 'passage_ko' hoặc chuỗi rỗng
    for k in ("passage", "passage_ko"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # Một số dữ liệu có thể để passage dạng list dòng:
    for k in ("passage", "passage_ko"):
        v = obj.get(k)
        if isinstance(v, list):
            items = []
            for it in v:
                if isinstance(it, str):
                    s = it.strip()
                    if s:
                        items.append(s)
                elif isinstance(it, dict):
                    t = (it.get("text") or it.get("utterance") or "").strip()
                    if t:
                        items.append(t)
            if items:
                return "\n".join(items)
    return ""

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
        if not norm and all(str(k).isdigit() for k in src.keys()):  # chuyển 1..n -> A..Z
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

# Map các cặp muốn gom trả cùng 1 record khi gọi 1 số bất kỳ trong cặp
_PAIR_MAP = {
    25: 26, 26: 25,
    27: 28, 28: 27,
    29: 30, 30: 29,
}

def _build_single_from_rec(level: str, danh_muc: str, rule: dict, data: dict, fallback_cau: int) -> dict:
    title   = (data.get("title")
               or _title_for_cau(int(data.get("question_no") or fallback_cau), fallback=data.get("title")))
    qtext   = _extract_question_text(data)
    choices = _normalize_choices(data)
    answer  = _normalize_answer(data, choices)

    raw_score = data.get("score")
    try:
        item_score = int(raw_score) if raw_score is not None else None
    except Exception:
        item_score = None

    fb_scores     = rule.get("fallback_score_by_cau", {})
    default_score = _default_score_for_level(_norm(level))
    score         = item_score if item_score is not None else fb_scores.get(int(data.get("question_no") or fallback_cau),
                                                                            default_score)

    # ==== Phân nhánh xuất: Listening -> dialog | Reading -> passage ====
    section = (data.get("section") or "").strip().lower()
    is_doc  = (section == "doc") or (_norm(danh_muc) == "reading")

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
        out["passage"] = _extract_passage(data)  # <== ĐỌC
    else:
        out["dialog"]  = _extract_dialog(data)   # <== NGHE
    return out

def _is_doc_like(d: dict) -> bool:
    section = (d.get("section") or "").strip().lower()
    t       = _norm(d.get("type"))
    return section == "doc" or t in ("reading", "doc")

def _build_pair_record(level: str, danh_muc: str, rule: dict, rec_a: dict, rec_b: dict) -> dict:
    data_a, data_b = rec_a["data"], rec_b["data"]
    item_a = _build_single_from_rec(level, danh_muc, rule, data_a, fallback_cau=25)
    item_b = _build_single_from_rec(level, danh_muc, rule, data_b, fallback_cau=26)

    # Xác định modality chung: Doc -> passage; Listening -> dialog
    doc_like = _is_doc_like(data_a) or _is_doc_like(data_b)

    if doc_like:
        shared_key = "passage"
        shared_val = _extract_passage(data_a) or _extract_passage(data_b) or ""
    else:
        shared_key = "dialog"
        shared_val = _extract_dialog(data_a) or _extract_dialog(data_b) or []

    # Tiêu đề nhóm theo dải
    lo = min(item_a.get("question_no") or 0, item_b.get("question_no") or 0)
    group_title = _title_for_cau(lo, fallback=data_a.get("title") or data_b.get("title"))

    # Bỏ field chung ở cấp item để tránh lặp
    strip_keys = {"dialog", "passage"}
    ia = {k: v for k, v in item_a.items() if k not in strip_keys}
    ib = {k: v for k, v in item_b.items() if k not in strip_keys}

    out = {
        "title":        group_title,
        "type":         data_a.get("type", danh_muc),
        "items":        [ia, ib],
        "question_no":  [ia["question_no"], ib["question_no"]],
        "score_total":  (ia.get("score", 0) + ib.get("score", 0)),
        shared_key:     shared_val,   # dialog hoặc passage chung
    }
    return out

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
      - ĐẶC BIỆT: nếu 'cau' thuộc cặp (25|26), (27|28), (29|30) → trả 1 record nhóm gồm cả 2 câu cùng bài.
    """
    cau = int(cau)
    lvl = _norm(level)
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
    _ensure_ready(dir_path)
    cache = _DIR_CACHE[dir_path]

    # === Nếu yêu cầu 1 số thuộc cặp → trả record nhóm gồm 2 câu cùng bài ===
    if cau in _PAIR_MAP and rule["cau_lo"] <= _PAIR_MAP[cau] <= rule["cau_hi"]:
        a, b = (cau, _PAIR_MAP[cau])
        hist_key = (user_key or "global", _norm(danh_muc), f"pair:{min(a,b)}|{max(a,b)}", str(dir_path))
        rec_a, rec_b = _choose_pair_by_group(cache, a, b, hist_key)
        return _build_pair_record(level, danh_muc, rule, rec_a, rec_b)

    # === Trường hợp đơn lẻ: ưu tiên qno, fallback theo score ===
    qno_required: bool = bool(rule.get("qno_required", False))
    fb_scores: Dict[int, int] = rule.get("fallback_score_by_cau", {})
    # default_score = _default_score_for_level(lvl)  # không còn cần ở đây; helper sẽ xử lý

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
            hist_key = (user_key or "global", _norm(danh_muc), f"score:{score_needed}", str(dir_path))
            chosen = _choose_nonrepeating(pool_score, hist_key)

    data = chosen["data"]
    # Build qua helper để Doc có "passage", Listening có "dialog"
    return _build_single_from_rec(level, danh_muc, rule, data, fallback_cau=cau)
