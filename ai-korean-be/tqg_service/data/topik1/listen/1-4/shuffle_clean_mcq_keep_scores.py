# shuffle_clean_mcq_keep_scores.py
# Xáo trộn vị trí đáp án + làm sạch + khử trùng lặp cho TOPIK MCQ
# Giữ score/mcp_* và REMAP theo trật tự mới.
# Bổ sung: loại các passage/prompt có dấu vết "prompt-artifact" (system/user/JSON…).

import os, re, json, random, hashlib
from typing import Dict, Any, Iterable, Tuple, List
from collections import Counter

# ==== Cấu hình ====
IN_PATH  = "/content/drive/MyDrive/topik_data/1-4/OUT/topik_mcq_1000.scored.jsonl"
OUT_PATH = "/content/drive/MyDrive/topik_data/1-4/OUT/topik_mcq_1000.shuffled.clean.jsonl"
DROP_LOG = "/content/drive/MyDrive/topik_data/1-4/OUT/topik_mcq_1000.shuffled.dropped.jsonl"

SEED = 13
MAX_CHOICE_LEN = 60

random.seed(SEED)

LETTER = {"A","B","C","D"}
ABC = ["A","B","C","D"]
NUM2LET = {"1":"A","2":"B","3":"C","4":"D", 1:"A",2:"B",3:"C",4:"D"}

# ==== I/O ====
def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            yield json.loads(line)

def write_jsonl(path: str, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")

# ==== Chuẩn hoá ====
def _to_letter_key(k):
    s = str(k).strip().upper()
    return s if s in LETTER else NUM2LET.get(s)

def normalize_item(ex: Dict[str,Any]) -> Tuple[Dict[str,Any], List[str]]:
    reasons=[]
    out = {}

    # prompt
    prompt = ex.get("prompt_ko") or ex.get("question") or "문맥에 맞는 대답을 고르십시오."
    out["prompt_ko"] = str(prompt).strip()

    # passage (nếu đã có)
    p = (ex.get("passage_ko") or "").strip()
    out["passage_ko"] = p if p else None

    # choices
    ch = ex.get("choices") or ex.get("options")
    if isinstance(ch, dict):
        canon={}
        for k,v in ch.items():
            lk=_to_letter_key(k)
            if not lk: continue
            canon[lk] = "" if v is None else str(v).strip()
        out["choices"] = {k: canon.get(k,"") for k in ABC}
    elif isinstance(ch, (list,tuple)) and len(ch)==4:
        out["choices"] = {k: (str(v).strip() if v else "") for k,v in zip(ABC, ch)}
    else:
        return None, ["choices_malformed"]

    # answer
    a = ex.get("answer")
    if isinstance(a, str):
        a = a.strip().upper()
        a = a if a in LETTER else NUM2LET.get(a, a)
    elif a in {1,2,3,4}:
        a = NUM2LET[a]
    out["answer"] = a

    # giữ score / mcp_*
    for k in ("score","mcp_probs","mcp_pred","mcp_margin","mcp_correct"):
        if k in ex:
            out[k] = ex[k]

    return out, reasons

# ==== Prompt-artifact filter ====
PROMPT_ARTIFACT_PATTERNS = [
    # vai trò/nhãn hội thoại
    r"^\s*(system|user|assistant)\b",            # đầu chuỗi
    # từ khoá hướng dẫn sinh
    r"오직\s*JSON만", r"JSON\s*금지", r"한국어\s*한\s*단락", r"지문/?대화", r"정답\s*JSON",
    r"TASK\s*:", r"FROM_PASSAGE", r"\bFULL\b", r"입력지문\s*:", r"VIEW\s*:", r"assistant_output",
    r"\bchoices\b", r"\banswer\b", r"\bprompt_ko\b",
    # mã/định dạng
    r"[{}]", r"```",
]

PROMPT_ARTIFACT_RE = re.compile("|".join(PROMPT_ARTIFACT_PATTERNS), flags=re.IGNORECASE)

def is_prompt_artifact_text(s: str) -> bool:
    if not s: return False
    txt = s.strip()
    # loại các chuỗi giống “system … user” dính liền
    if re.search(PROMPT_ARTIFACT_RE, txt):
        return True
    # các vết nối role
    if "system" in txt.lower() and "user" in txt.lower():
        return True
    return False

# ==== Validate (bao gồm artifact) ====
def is_valid(item: Dict[str,Any]) -> Tuple[bool, List[str]]:
    reasons=[]
    if not isinstance(item.get("choices"), dict) or set(item["choices"].keys()) != LETTER:
        return False, ["choices_not_A_B_C_D"]
    if item.get("answer") not in LETTER:
        return False, ["answer_invalid"]

    vals = [item["choices"][k].strip() for k in ABC]
    if any(not v for v in vals):
        return False, ["empty_choice"]
    if any(len(v) > MAX_CHOICE_LEN for v in vals):
        reasons.append("choice_too_long")
    if len(set(vals)) < 4:
        reasons.append("duplicate_choice_texts")

    # artifact ở passage/prompt/choices
    p = (item.get("passage_ko") or "").strip()
    if p and is_prompt_artifact_text(p):
        return False, ["artifact_passage"]

    q = (item.get("prompt_ko") or "").strip()
    if is_prompt_artifact_text(q):
        return False, ["artifact_prompt"]

    if any(is_prompt_artifact_text(v) for v in vals):
        return False, ["artifact_choice"]

    return (len(reasons)==0), reasons

# ==== Dedup chữ ký không phụ thuộc vị trí A-D ====
def signature_unordered(item: Dict[str,Any]) -> str:
    p = (item.get("passage_ko") or "").strip()
    q = (item.get("prompt_ko") or "").strip()
    ch = item.get("choices", {})
    pool = sorted([str(ch.get(k,"")).strip() for k in ABC])
    key = "||".join([p, q, *pool])
    return hashlib.md5(key.encode("utf-8")).hexdigest()

# ==== Remap MCP theo hoán vị ====
def remap_scores_after_permutation(item: Dict[str,Any], old_to_new: Dict[str,str]) -> None:
    probs = item.get("mcp_probs", None)
    if isinstance(probs, dict):
        new_probs = { old_to_new.get(k, k): float(v) for k,v in probs.items() if k in LETTER }
        item["mcp_probs"] = {k: float(new_probs.get(k, 0.0)) for k in ABC}
    elif isinstance(probs, (list,tuple)) and len(probs)==4:
        tmp = dict(zip(ABC, [float(x) for x in probs]))
        new_probs = { old_to_new.get(k, k): tmp[k] for k in ABC }
        item["mcp_probs"] = {k: float(new_probs.get(k, 0.0)) for k in ABC}

    pred = item.get("mcp_pred", None)
    if isinstance(pred, str):
        sp = pred.strip().upper()
        if sp in LETTER:
            item["mcp_pred"] = old_to_new.get(sp, sp)

    if "mcp_pred" in item and "answer" in item:
        item["mcp_correct"] = bool(item["mcp_pred"] == item["answer"])
    # mcp_margin & score: giữ nguyên

def shuffle_item(item: Dict[str,Any]) -> Dict[str,Any]:
    labels = ABC[:]
    perm = random.sample(labels, k=4)   # new_label lấy text từ old_label=perm[i]

    old = item["choices"]
    new_choices = {nl: old[ol] for nl, ol in zip(labels, perm)}

    new_to_old = {nl: ol for nl, ol in zip(labels, perm)}
    old_to_new = {ol: nl for nl, ol in new_to_old.items()}

    new_ans = old_to_new[item["answer"]]

    new_item = {
        "prompt_ko": item["prompt_ko"],
        "choices": new_choices,
        "answer": new_ans
    }
    if item.get("passage_ko"):
        new_item["passage_ko"] = item["passage_ko"]

    for k in ("score","mcp_probs","mcp_pred","mcp_margin","mcp_correct"):
        if k in item:
            new_item[k] = item[k]

    remap_scores_after_permutation(new_item, old_to_new)
    return new_item

# ==== Main ====
def main():
    kept, dropped = [], []
    seen = set()

    # 1) normalize + validate + dedup
    for raw in read_jsonl(IN_PATH):
        item, _ = normalize_item(raw)
        if item is None:
            dropped.append({"reason":["normalize_failed"], "raw": raw})
            continue
        ok, reasons = is_valid(item)
        if not ok:
            dropped.append({"reason": reasons, "raw": raw})
            continue
        sig = signature_unordered(item)
        if sig in seen:
            dropped.append({"reason":["duplicate_signature"], "raw": raw})
            continue
        seen.add(sig)
        kept.append(item)

    # 2) shuffle + giữ điểm (đã remap)
    shuffled = [shuffle_item(it) for it in kept]

    # 3) ghi file
    write_jsonl(OUT_PATH, shuffled)
    write_jsonl(DROP_LOG, dropped)

    # 4) báo cáo
    before_ans = Counter([it["answer"] for it in kept])
    after_ans  = Counter([it["answer"] for it in shuffled])
    print(f"Input: {IN_PATH}")
    print(f"Kept: {len(kept)} | Dropped: {len(dropped)}")
    print(f"Saved shuffled: {OUT_PATH}")
    print("Answer distribution before shuffle:", dict(before_ans))
    print("Answer distribution after  shuffle:", dict(after_ans))

if __name__ == "__main__":
    main()
