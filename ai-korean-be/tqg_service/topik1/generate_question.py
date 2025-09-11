# topik_service.py
import json, os, time, random, hashlib, threading, re
from pathlib import Path
from collections import deque, defaultdict
from typing import Dict, Optional, List

# ====== CẤU HÌNH DỮ LIỆU ======
TOPIK14_LISTEN_DIR = Path(os.getenv("TOPIK14_LISTEN_DIR", "data/topik1/listen/1-4/OUT")).resolve()
TARGET_SCORE_BY_CAU = {1: 4, 2: 4, 3: 3, 4: 3}

# ====== TITLES MAPPING (theo khoảng câu) ======
_TITLES_RANGES: List[tuple] = [
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

# ====== CACHE TOÀN CỤC ======
_CACHE_LOCK = threading.RLock()
_TOPIK14_LISTEN_CACHE = {
    "loaded_at": 0.0,
    "files": [],                 # List[Path]
    "mtimes": {},                # {Path: float}
    "by_score": {3: [], 4: []},  # mỗi phần tử: {"id": str, "data": dict}
}

# Lịch sử đã trả (tránh trùng) per (user_key, danh_muc, score)
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

def _needs_refresh() -> bool:
    if not _TOPIK14_LISTEN_CACHE["files"]:
        return True
    current = list(TOPIK14_LISTEN_DIR.glob("*.jsonl"))
    if set(current) != set(_TOPIK14_LISTEN_CACHE["files"]):
        return True
    for p in current:
        m = p.stat().st_mtime
        if _TOPIK14_LISTEN_CACHE["mtimes"].get(p) != m:
            return True
    return False

def _refresh_cache():
    with _CACHE_LOCK:
        by_score = {3: [], 4: []}
        files = list(TOPIK14_LISTEN_DIR.glob("*.jsonl"))
        mtimes = {}
        for p in files:
            mtimes[p] = p.stat().st_mtime
            for obj in _load_jsonl(p):
                # chuẩn hóa score
                sc = obj.get("score", None)
                try:
                    sc = int(sc)
                except Exception:
                    continue
                if sc not in (3, 4):
                    continue
                qid = obj.get("id") or _stable_id(obj)
                by_score[sc].append({"id": qid, "data": obj})

        random.shuffle(by_score[3])
        random.shuffle(by_score[4])

        _TOPIK14_LISTEN_CACHE["by_score"] = by_score
        _TOPIK14_LISTEN_CACHE["files"] = files
        _TOPIK14_LISTEN_CACHE["mtimes"] = mtimes
        _TOPIK14_LISTEN_CACHE["loaded_at"] = time.time()

def warm_topik14_listen_cache():
    if TOPIK14_LISTEN_DIR.exists():
        _refresh_cache()

def _choose_nonrepeating(danh_muc: str, score: int, user_key: str) -> dict:
    with _CACHE_LOCK:
        pool = _TOPIK14_LISTEN_CACHE["by_score"].get(score, [])
        if not pool:
            raise ValueError(f"Không có câu hỏi với score={score} trong {TOPIK14_LISTEN_DIR}")

        hist_key = (user_key, danh_muc, score)
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
    # ưu tiên các field quen thuộc
    return (
        obj.get("question")
        or obj.get("question_ko")
        or obj.get("prompt")
        or obj.get("prompt_ko")
        or ""
    )

def _extract_dialog(obj: dict) -> List[dict]:
    """
    Lấy dialog:
      - Nếu đã có obj['dialogue'] dạng list[{'speaker','text'}] -> dùng luôn
      - Nếu có 'passage_ko':
          + string: tách từng dòng; nếu có '남자:' / '여자:'... thì split speaker:text
                    nếu không, speaker="지문"
          + list[str]: mỗi phần tử thành một turn speaker="지문"
          + list[dict]: nếu đã có speaker/text thì chuẩn hóa key
    """
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

    # fallback
    return []

def _normalize_choices(obj: dict) -> Dict[str, str]:
    # hỗ trợ 'choices' hoặc 'options', list hoặc dict
    src = obj.get("choices")
    if src is None:
        src = obj.get("options")
    if src is None:
        return {}

    if isinstance(src, dict):
        # chuẩn hóa key thành A/B/C/D...
        norm = {}
        for k, v in src.items():
            if not isinstance(v, str):
                continue
            key = str(k).strip().upper()
            if len(key) == 1 and "A" <= key <= "Z":
                norm[key] = v
        # nếu key là "1","2"... thì map sang A,B...
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

    # nếu đã là chữ A/B/C/D
    if isinstance(ans, str):
        up = ans.strip().upper()
        if up in choices:
            return up
        # nếu là nội dung text, map ngược
        for k, v in choices.items():
            if v.strip() == ans.strip():
                return k
        # nếu là "1","2"...
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
        # thử 1-based trước
        if 1 <= ans <= len(choices):
            return letters[ans - 1]
        # rồi 0-based
        if 0 <= ans < len(choices):
            return letters[ans]
        return "A" if choices else ""

    return "A" if choices else ""

# ====== API CHÍNH (CHỈ DATASET, KHÔNG GỌI AI) ======
class DataNotFoundError(Exception): ...
class BadRequestError(Exception): ...

def generate_topik_question(
    danh_muc: str,
    cau: int,
    *,
    user_key: Optional[str] = "global",
) -> dict:
    """
    Câu 1–4: lấy từ dataset OUT, 1–2 -> score=4, 3–4 -> score=3, tránh trùng theo user_key.
    Trả về:
      {
        "title": <theo mapping nếu 1–4>,
        "type": data.get("type", danh_muc),
        "question": <chuỗi hỏi>,
        "dialog":  list[{"speaker","text"}] lấy từ passage_ko/dialogue,
        "choices": {"A": "...", ...},
        "answer":  "A"/"B"/...,
        "score":   int
      }
    """
    if not (1 <= int(cau) <= 4):
        raise BadRequestError("Hiện chỉ hỗ trợ câu 1–4 từ dataset.")

    if not TOPIK14_LISTEN_DIR.exists():
        raise DataNotFoundError(f"Không thấy thư mục: {TOPIK14_LISTEN_DIR}")

    if _needs_refresh():
        _refresh_cache()

    target_score = TARGET_SCORE_BY_CAU[int(cau)]
    chosen = _choose_nonrepeating(danh_muc=danh_muc, score=target_score, user_key=user_key or "global")
    data = chosen["data"]

    # build fields
    title = _title_for_cau(int(cau), fallback=data.get("title"))
    qtext = _extract_question_text(data)
    dialog = _extract_dialog(data)
    choices = _normalize_choices(data)
    answer = _normalize_answer(data, choices)
    score  = int(data.get("score", target_score))

    return {
        "title": title,
        "type":  data.get("type", danh_muc),
        "question": qtext or "",      # “không có để trống”
        "dialog":  dialog,
        "choices": choices,
        "answer":  answer,
        "score":   score,
    }
