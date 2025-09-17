# =========================
# TOPIK 5–6 GENERATOR INFER
# - load base + LoRA adapter
# - sinh từ Controls (topic, question_no) => full JSON (dialogue + options + answer)
# - validator + generate-k-candidates -> pick best
# =========================

import os, json, random, re, time
from typing import Tuple, Optional, Dict, Any, List
from collections import Counter

# ---------- CONFIG ----------
BASE_MODEL  = "meta-llama/Meta-Llama-3-8B-Instruct"  # hoặc: "microsoft/Phi-3-mini-4k-instruct", "Qwen2.5-3B-Instruct"
ADAPTER_DIR = "/content/drive/MyDrive/topik1/listen/5-6/output/topik-56-llama3-8b-lora"  # đổi theo nơi bạn save adapter khi train
CACHE_DIR   = "/content/hf-cache"  # cache tạm ở /content, không chiếm Drive

# decode mặc định (có thể chỉnh khi gọi gen_best)
GEN_K                 = 5          # số ứng viên mỗi lần
MAX_ATTEMPTS          = 2          # số vòng re-try nếu chưa có ứng viên hợp lệ
MAX_NEW_TOKENS        = 240
TEMPERATURES_PER_TRY  = [0.7, 0.6]  # vòng 1 0.7, vòng 2 0.6
TOP_P                 = 0.9
REPETITION_PENALTY    = 1.12

# xuất batch
SAVE_JSONL_PATH = "/content/topik56_generated.jsonl"  # file output nếu chạy batch

# ---------- ENV / CACHE ----------
import pathlib
pathlib.Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# ---------- LOAD MODEL ----------
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def load_model(base_model: str, adapter_dir: str):
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, cache_dir=CACHE_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    base = AutoModelForCausalLM.from_pretrained(
        base_model, cache_dir=CACHE_DIR, quantization_config=bnb, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return tok, model

tok, model = load_model(BASE_MODEL, ADAPTER_DIR)

# ---------- PROMPT (KHỚP VỚI TRAIN) ----------
INSTR = """### Instruction:
당신은 TOPIK I (초급) 5~6 유형 문제를 '처음부터' 생성하는 모델입니다.
아래의 Controls(주제/번호)에 맞춰, 다음 JSON 한 개만 출력하세요:
{
  "type": "Nghe_Loi_Tiep",
  "section": "Nghe",
  "level": "1",
  "title": "[5~6] 다음을 듣고 <보기>와 같이 이어지는 말을 고르십시오.",
  "question": "이어지는 말을 고르십시오.",
  "explanation": "",
  "score": 4,
  "options": {"A":"...", "B":"...", "C":"...", "D":"..."},
  "answer": "A|B|C|D",
  "dialogue": [ { "speaker":"남자|여자", "text":"..." } ],
  "question_no": 5|6
}
제한:
- TOPIK I 일상 어휘.
- "dialogue"는 한 줄(한 화자)만.
- options 4개는 서로 달라야 함.
- answer는 options의 키 중 하나.
- JSON 외의 설명/텍스트는 출력하지 마세요.
"""

PROMPT_TMPL = """<|begin_of_text|>
{instr}

### Controls:
topic: {topic}
question_no: {qno}

### Response:
"""

def build_prompt(topic: str, qno: int) -> str:
    return PROMPT_TMPL.format(instr=INSTR, topic=topic, qno=qno)

# ---------- VALIDATOR ----------
NEED_KEYS = {"type","section","level","title","question","explanation","score","options","answer","dialogue","question_no"}

def is_korean_text(s: str) -> bool:
    # kiểm tra có ký tự Hangul cơ bản
    return bool(re.search(r"[\uac00-\ud7af]", s or ""))

def validate_json(s: str) -> Tuple[bool, Any]:
    try:
        obj = json.loads(s)
    except Exception:
        return False, "parse"
    miss = NEED_KEYS - obj.keys()
    if miss: return False, f"missing:{sorted(list(miss))}"
    if not isinstance(obj.get("options"), dict): return False, "options_not_dict"
    if set(obj["options"].keys()) != {"A","B","C","D"}: return False, "options_keys"
    if obj.get("answer") not in {"A","B","C","D"}: return False, "answer_label"
    dlg = obj.get("dialogue")
    if not isinstance(dlg, list) or len(dlg) != 1: return False, "dialogue_len"
    spk = dlg[0].get("speaker")
    if spk not in ("남자","여자"): return False, "speaker"
    if obj.get("question_no") not in (5,6): return False, "question_no"
    # options phải khác nhau & không rỗng
    vals = [str(v).strip() for v in obj["options"].values()]
    if any(len(v)==0 for v in vals): return False, "empty_option"
    if len(set(vals)) < 4: return False, "duplicate_options"
    # length & ngôn ngữ cơ bản
    if not is_korean_text(dlg[0].get("text","")): return False, "dialogue_not_korean"
    if len(dlg[0].get("text","")) > 40: return False, "dialogue_too_long"
    for v in vals:
        if not is_korean_text(v): return False, "option_not_korean"
        if len(v) > 30: return False, "option_too_long"
    return True, obj

# ---------- SCORER (đơn giản, có thể điều chỉnh) ----------
def score_item(obj: Dict[str,Any]) -> float:
    score = 0.0
    # thưởng tính gọn gàng
    dtext = obj["dialogue"][0]["text"].strip()
    L = len(dtext)
    if 6 <= L <= 28: score += 2.0
    elif 3 <= L <= 34: score += 1.0

    # thưởng options gọn & khác nhau (đã đảm bảo khác nhau ở validator)
    for v in obj["options"].values():
        lv = len(v.strip())
        if 3 <= lv <= 20: score += 0.5

    # nhẹ tay cân bằng question_no & answer (tăng chút cho 6 để đa dạng)
    score += 0.2 if obj["question_no"] == 6 else 0.0
    # tránh thiên vị 1 đáp án quá nhiều: không thể biết batch ở đây, nên không phạt ở mức cá lẻ

    return score

# ---------- GENERATE ----------
def generate_raw(topic: str, qno: int, seed: int, temperature: float, top_p: float) -> str:
    import torch
    torch.manual_seed(seed)
    prompt = build_prompt(topic, qno)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=REPETITION_PENALTY,
        eos_token_id=tok.eos_token_id,
    )
    full = tok.decode(out[0], skip_special_tokens=True)
    body = full.split("### Response:")[-1].strip()
    # nếu model lỡ in thêm text sau JSON, cắt thô theo "}\n" gần nhất
    if body.count("{") > 0 and body.count("}") > 0:
        try:
            end = body.rfind("}")
            if end != -1: body = body[:end+1]
        except Exception:
            pass
    return body

def gen_best(topic: str, qno: int, k:int=GEN_K, max_attempts:int=MAX_ATTEMPTS) -> Tuple[Optional[Dict[str,Any]], List[Dict[str,Any]]]:
    """Trả về: (obj_tốt_nhất hoặc None, danh sách logs ứng viên)"""
    logs: List[Dict[str,Any]] = []
    base_seed = random.randint(1_000_000, 9_999_999)

    for attempt in range(max_attempts):
        temp = TEMPERATURES_PER_TRY[min(attempt, len(TEMPERATURES_PER_TRY)-1)]
        candidates = []
        for i in range(k):
            seed = base_seed + attempt*100 + i
            raw = generate_raw(topic, qno, seed=seed, temperature=temp, top_p=TOP_P)
            ok, obj_or_reason = validate_json(raw)
            log = {"attempt": attempt+1, "seed": seed, "temperature": temp, "valid": ok}
            if ok:
                obj = obj_or_reason
                log["score"] = score_item(obj)
                log["obj"] = obj
                candidates.append( (log["score"], obj, log) )
            else:
                log["reason"] = obj_or_reason
                log["raw"] = raw
            logs.append(log)

        # chọn best nếu có
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            best = candidates[0][1]
            return best, logs

        # nếu chưa có valid: giảm nhiệt độ và thử lại (vòng sau)

    # không có ứng viên hợp lệ
    return None, logs

# ---------- DEMO 1: sinh 1 item ----------
best, logs = gen_best(topic="shopping", qno=5, k=5, max_attempts=2)
print("VALID:", best is not None)
if best:
    print(json.dumps(best, ensure_ascii=False, indent=2))
else:
    # in một vài log lỗi để biết vì sao rớt
    for lg in logs[:3]:
        print(lg.get("reason"), "seed=", lg.get("seed"))

# ---------- DEMO 2: sinh batch N items & lưu JSONL ----------
def generate_batch(
    topics: List[str],
    per_topic:int = 20,
    qno_ratio:Tuple[int,int]=(1,1),
    out_path:str = SAVE_JSONL_PATH,
    k_each:int=GEN_K,
    attempts:int=MAX_ATTEMPTS,
):
    total = 0
    ok = 0
    ans_dist = Counter()
    qno_dist = Counter()
    spk_dist = Counter()
    with open(out_path, "w", encoding="utf-8") as f:
        for topic in topics:
            # xen kẽ qno theo tỉ lệ qno_ratio
            qnos = [5]*qno_ratio[0] + [6]*qno_ratio[1]
            for i in range(per_topic):
                qno = qnos[i % len(qnos)]
                obj, _ = gen_best(topic, qno, k=k_each, max_attempts=attempts)
                total += 1
                if obj:
                    ok += 1
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    ans_dist[obj["answer"]] += 1
                    qno_dist[obj["question_no"]] += 1
                    spk_dist[obj["dialogue"][0]["speaker"]] += 1
    print(f"[BATCH] saved to {out_path}")
    print(f"[BATCH] valid {ok}/{total} = {ok/total:.1%}")
    print("[DIST] answer:", dict(ans_dist))
    print("[DIST] qno:", dict(qno_dist))
    print("[DIST] speaker:", dict(spk_dist))

# Ví dụ chạy batch (bỏ comment để chạy):
# topics_list = ["shopping","school","work","restaurant","transport","health","bank_post","home","tech","social","travel_hotel","weather"]
# generate_batch(topics_list, per_topic=10, qno_ratio=(1,1), out_path=SAVE_JSONL_PATH, k_each=5, attempts=2)
# Sau khi chạy xong, tải file:
# from google.colab import files; files.download(SAVE_JSONL_PATH)
