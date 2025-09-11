# ===== TOPIK MCP Cleaner + Batch-Normalized Scoring (no GEN) =====
# - Accepts BOTH schemas:
#   (A) raw TOPIK: {"question", "options": {"A":...}, "answer": "A", "dialogue":[{"speaker","text"},...], ...}
#   (B) canonical: {"passage_ko","prompt_ko","choices":{"A"...},"answer":"A"}
# - Cleans -> Normalizes -> Filters -> Computes MCP probabilities -> margin
# - Batch-normalized scoring: assigns 3 points to top TARGET_PCT_3 of margins (easier), 4 points to the rest.

import os, re, json, math, statistics, hashlib, unicodedata
from collections import Counter
from typing import Dict, Any, Iterable, Tuple, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import login

# ----------------- CONFIG -----------------
BASE         = "meta-llama/Meta-Llama-3-8B-Instruct"
MCP_ADAPTER  = "nguyentieuat/topik-mcq-llama3-8b-lora"

INPUT_JSONL  = "/kaggle/working/topik_mcq_1000.jsonl"        # <== đổi cho đúng
OUTPUT_JSONL = "/kaggle/working/topik_mcq_1000.scored.jsonl" # đã clean + score
DROPPED_JSONL= "/kaggle/working/topik_mcq_1000.dropped.jsonl"# log bị loại

# Batch normalization: tỉ lệ câu 3 điểm mong muốn
TARGET_PCT_3 = 0.40   # 40% câu dễ (margin cao) -> 3 điểm; còn lại -> 4 điểm

# Cleaning thresholds (có thể chỉnh)
MIN_KO_RATIO_PASSAGE = 0.40   # tối thiểu 40% ký tự Hangul trong passage (nếu có)
MIN_KO_RATIO_CHOICE  = 0.40
MAX_CHOICE_LEN       = 30     # mỗi lựa chọn <= 30 ký tự
MAX_LEN_VARIANCE     = 30     # |max_len - min_len| giữa 4 lựa chọn
BAN_DUP_CHOICE       = True   # loại nếu trùng lựa chọn
MIN_PASSAGE_LEN      = 5
MAX_PASSAGE_LEN      = 160

# Login HF nếu base là gated (an toàn cứ để vậy)
try: login()
except Exception: pass

# Tối ưu CUDA chunking (không bắt buộc)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True

# ----------------- I/O -----------------
def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            yield json.loads(line)

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")

# ----------------- Utils: text & schema -----------------
HANGUL_RE = re.compile(r"[가-힣]")

def ko_ratio(s: str) -> float:
    if not s: return 0.0
    s = str(s)
    ko = len(HANGUL_RE.findall(s))
    # đếm chữ cái/số/ký tự có ý nghĩa
    tokens = [ch for ch in s if not unicodedata.category(ch).startswith("Z")]  # bỏ space
    tot = max(1, len(tokens))
    return ko / tot

def sig_item(passage: str, prompt: str, ch: Dict[str,str]) -> str:
    key = "|".join([
        (passage or "").strip(),
        (prompt or "").strip(),
        *(str(ch.get(k,"")).strip() for k in ("A","B","C","D"))
    ])
    return hashlib.md5(key.encode("utf-8")).hexdigest()

LETTER = {"A","B","C","D"}
NUM2LET = {"1":"A","2":"B","3":"C","4":"D", 1:"A",2:"B",3:"C",4:"D"}

def _to_letter_key(k):
    s = str(k).strip().upper()
    return s if s in LETTER else NUM2LET.get(s)

def build_passage_from_dialogue(dlg: List[Dict[str,str]]) -> str:
    lines=[]
    for turn in dlg or []:
        spk = (turn.get("speaker") or "").strip()
        txt = (turn.get("text") or "").strip()
        if not txt: continue
        if spk:
            lines.append(f"{spk}: {txt}")
        else:
            lines.append(txt)
    # 1-3 dòng là ổn, cắt gọn
    if len(lines) > 3: lines = lines[:3]
    return " ".join(lines).strip()

# ----------------- Normalize schemas -> canonical -----------------
def to_canonical(ex: Dict[str,Any]) -> Tuple[Dict[str,Any], List[str]]:
    """Return (item, reasons). item has: passage_ko?, prompt_ko, choices{A..D}, answer"""
    reasons=[]
    out={}

    # Case A: raw TOPIK
    if "options" in ex and isinstance(ex.get("options"), dict):
        # passage từ dialogue (nếu có)
        passage = None
        if isinstance(ex.get("dialogue"), list) and ex["dialogue"]:
            passage = build_passage_from_dialogue(ex["dialogue"])
        # prompt/question
        prompt = ex.get("question") or "문맥에 맞는 대답을 고르십시오."
        # choices
        ch_raw = ex["options"]
        canon={}
        for k,v in ch_raw.items():
            lk=_to_letter_key(k)
            if lk: canon[lk]=(str(v).strip() if v is not None else "")
        # answer
        ans = ex.get("answer")
        if isinstance(ans, str):
            ans = ans.strip().upper()
            ans = ans if ans in LETTER else NUM2LET.get(ans, ans)
        elif ans in {1,2,3,4}:
            ans = NUM2LET[ans]

        out = {
            "passage_ko": passage,
            "prompt_ko": str(prompt).strip(),
            "choices": {k: canon.get(k,"") for k in ["A","B","C","D"]},
            "answer": ans
        }
        return out, reasons

    # Case B: canonical đã có
    prompt = ex.get("prompt_ko") or "문맥에 맞는 대답을 고르십시오."
    ch_raw = ex.get("choices", {})
    canon={}
    if isinstance(ch_raw, dict):
        for k,v in ch_raw.items():
            lk=_to_letter_key(k)
            if lk: canon[lk]=(str(v).strip() if v is not None else "")
        if set(canon.keys()) != LETTER and all(str(k) in {"1","2","3","4"} for k in ch_raw.keys()):
            ordered=[ch_raw.get(i, ch_raw.get(str(i),"")) for i in [1,2,3,4]]
            canon={L: (str(v).strip() if v is not None else "") for L,v in zip(["A","B","C","D"], ordered)}
    elif isinstance(ch_raw,(list,tuple)) and len(ch_raw)==4:
        canon={L: (str(v).strip() if v is not None else "") for L,v in zip(["A","B","C","D"], ch_raw)}

    ans = ex.get("answer")
    if isinstance(ans, str):
        ans = ans.strip().upper()
        ans = ans if ans in LETTER else NUM2LET.get(ans, ans)
    elif ans in {1,2,3,4}:
        ans = NUM2LET[ans]

    out = {
        "passage_ko": (ex.get("passage_ko") or None),
        "prompt_ko": str(prompt).strip(),
        "choices": {k: canon.get(k,"") for k in ["A","B","C","D"]},
        "answer": ans
    }
    return out, reasons

# ----------------- Cleaning rules -----------------
def clean_item(item: Dict[str,Any]) -> Tuple[bool, List[str]]:
    """Return (keep?, reasons_if_dropped)"""
    probs=[]
    reasons=[]

    # 1) keys + answer
    if not isinstance(item.get("choices"), dict) or set(item["choices"].keys())!=LETTER:
        return False, ["choices_not_A_B_C_D"]
    if item.get("answer") not in LETTER:
        return False, ["answer_invalid"]

    # 2) lengths + variance
    vals = [item["choices"][k].strip() for k in ["A","B","C","D"]]
    if any(not v for v in vals):
        return False, ["empty_choice"]
    if any(len(v) > MAX_CHOICE_LEN for v in vals):
        return False, ["choice_too_long"]
    L = [len(v) for v in vals]
    if max(L) - min(L) > MAX_LEN_VARIANCE:
        return False, ["len_variance_too_high"]

    # 3) dup choices
    if BAN_DUP_CHOICE and len(set(vals)) < 4:
        return False, ["duplicate_choices"]

    # 4) Korean ratio on choices
    if any(ko_ratio(v) < MIN_KO_RATIO_CHOICE for v in vals):
        return False, ["low_ko_ratio_choice"]

    # 5) passage constraints (if present)
    p = (item.get("passage_ko") or "").strip()
    if p:
        if len(p) < MIN_PASSAGE_LEN or len(p) > MAX_PASSAGE_LEN:
            return False, ["passage_len_out_of_range"]
        if ko_ratio(p) < MIN_KO_RATIO_PASSAGE:
            return False, ["low_ko_ratio_passage"]

    return True, []

# ----------------- Load MCP only -----------------
def load_base():
    try:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        m = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto").eval()
        print("✓ Base loaded in 4-bit.")
        return m
    except Exception as e:
        print("[4-bit failed] → fallback FP16 sharded.", type(e).__name__, str(e)[:180])

    max_mem = {i: "7GiB" for i in range(torch.cuda.device_count())}
    max_mem["cpu"] = "32GiB"
    os.makedirs("./offload", exist_ok=True)
    m = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.float16, device_map="auto",
        max_memory=max_mem, low_cpu_mem_usage=True, offload_folder="./offload"
    ).eval()
    try: m.config.attn_implementation = "sdpa"
    except: pass
    print("✓ Base loaded in FP16 sharded.")
    return m

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
base = load_base()

# tránh re-dispatch khi gắn adapter 4-bit
_prev_map = getattr(base, "hf_device_map", None)
try: delattr(base, "hf_device_map")
except Exception: base.hf_device_map = None
mcp = PeftModel.from_pretrained(base, MCP_ADAPTER, is_trainable=False).eval()
if _prev_map is not None:
    base.hf_device_map = _prev_map
print("✅ MCP adapter attached.")

ABC = ["A","B","C","D"]
AB_IDS = [tok.encode(x, add_special_tokens=False)[0] for x in ABC]
print("Token IDs A/B/C/D:", AB_IDS)

def build_clf_prompt(item: Dict[str, Any]) -> str:
    passage = (item.get("passage_ko") or "").strip()
    q = (item.get("prompt_ko") or "문맥에 맞는 대답을 고르십시오.").strip()
    ch = item["choices"]
    lines=[]
    if passage: lines.append(f"지문:\n{passage}\n")
    lines.append(f"문제: {q}")
    lines.append(f"보기:\nA) {ch['A']}\nB) {ch['B']}\nC) {ch['C']}\nD) {ch['D']}")
    lines.append("정답만 A/B/C/D 중 하나로 출력하세요.")
    return "\n".join(lines)

@torch.no_grad()
def probs_abcd_forward(model, prompt_ids):
    out = model(**prompt_ids, use_cache=False)         # logits: (1, T, vocab)
    last = out.logits[:, -1, :]                        # (1, vocab)
    probs = torch.softmax(last[0, AB_IDS], dim=-1)     # normalize over A-D
    return probs.tolist()

def margin_pred_for_item(item: Dict[str,Any]) -> Tuple[float, str, Dict[str,float]]:
    msgs=[{"role":"system","content":"너는 객관식 정답 선택기이다. 반드시 A/B/C/D 중 하나만 출력해라."},
          {"role":"user","content": build_clf_prompt(item)}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(mcp.device)
    ps = probs_abcd_forward(mcp, ids)                  # [pA,pB,pC,pD]
    idx = int(max(range(4), key=lambda i: ps[i]))
    pred = ABC[idx]
    p_gold = ps[ABC.index(item["answer"])]
    p_others = max(p for i,p in enumerate(ps) if ABC[i]!=item["answer"])
    margin = float(p_gold - p_others)
    return margin, pred, dict(zip(ABC, [float(x) for x in ps]))

# ----------------- Pipeline -----------------
def run_clean_and_score():
    seen = set()
    kept = []
    dropped = []

    # 1) normalize + clean + dedup
    for raw in read_jsonl(INPUT_JSONL):
        item, _ = to_canonical(raw)

        # signature for dedup
        sig = sig_item(item.get("passage_ko",""), item.get("prompt_ko",""), item.get("choices",{}))
        if sig in seen:
            dropped.append({"reason":["dup_signature"], "raw": raw})
            continue
        keep, reasons = clean_item(item)
        if not keep:
            dropped.append({"reason": reasons, "raw": raw})
            continue
        seen.add(sig)
        kept.append(item)

    if not kept:
        print("No valid examples after cleaning!")
        write_jsonl(DROPPED_JSONL, dropped)
        return

    # 2) MCP margin + prediction
    for it in kept:
        m, pred, probs = margin_pred_for_item(it)
        it["mcp_probs"]   = probs
        it["mcp_margin"]  = round(m, 6)
        it["mcp_pred"]    = pred
        it["mcp_correct"] = bool(pred == it["answer"])

    # 3) Batch-normalized scoring
    margins = [it["mcp_margin"] for it in kept]
    # threshold so that ~TARGET_PCT_3 get 3 points (higher margin -> easier)
    import numpy as np
    q = float(np.quantile(margins, 1.0 - TARGET_PCT_3))
    for it in kept:
        it["score"] = 3 if it["mcp_margin"] >= q else 4
    print(f"[Batch normalize] TARGET_PCT_3={TARGET_PCT_3:.2f} → threshold margin={q:.4f}")

    # 4) Write outputs
    write_jsonl(OUTPUT_JSONL, kept)
    write_jsonl(DROPPED_JSONL, dropped)

    # 5) Summary
    cnt_ans = Counter([it["answer"] for it in kept])
    cnt_pred= Counter([it["mcp_pred"] for it in kept])
    cnt_score= Counter([it["score"] for it in kept])
    acc = sum(1 for it in kept if it["mcp_correct"]) / max(1,len(kept))
    mean_margin = statistics.mean(margins)
    print(f"Kept: {len(kept)} | Dropped: {len(dropped)}")
    print(f"Accuracy vs gold: {acc:.3f} | mean margin: {mean_margin:.3f}")
    print("Answer distribution:", dict(cnt_ans))
    print("Pred distribution:  ", dict(cnt_pred))
    print("Score distribution: ", dict(cnt_score))

if __name__ == "__main__":
    run_clean_and_score()
