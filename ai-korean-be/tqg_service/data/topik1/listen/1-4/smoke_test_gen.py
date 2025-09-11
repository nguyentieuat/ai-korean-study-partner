# ==== Load TOPIK generator adapter từ Hugging Face + smoke test ====
import os, json, torch, re
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel

BASE    = "meta-llama/Meta-Llama-3-8B-Instruct"          # base gated -> cần login HF để có quyền
ADAPTER = "nguyentieuat/topik-gen-llama3-8b-lora"        # <== adapter trên HF (repo_id)

# (khuyên) login để đảm bảo quyền vào base model
from huggingface_hub import login
try:
    login()  # dán token nếu cần
except Exception:
    pass

# Mẹo giảm phân mảnh VRAM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True

# ---- Load tokenizer ----
tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# ---- Load base model với 3 tầng fallback ----
def load_base():
    # 1) Thử 4-bit (nhẹ nhất)
    try:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        m = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto").eval()
        print("✓ Loaded base in 4-bit.")
        return m
    except Exception as e:
        print("[4-bit failed] → fallback FP16 sharded. Err:", type(e).__name__, str(e)[:200])

    # 2) FP16 sharded (balanced_low_0 → auto)
    max_mem = {i: "7GiB" for i in range(torch.cuda.device_count())}
    max_mem["cpu"] = "32GiB"
    os.makedirs("./offload", exist_ok=True)

    for dm in ("balanced_low_0", "auto"):
        try:
            m = AutoModelForCausalLM.from_pretrained(
                BASE,
                torch_dtype=torch.float16,
                device_map=dm,
                max_memory=max_mem,
                low_cpu_mem_usage=True,     # bắt buộc khi có device_map
                offload_folder="./offload",
            ).eval()
            try: m.config.attn_implementation = "sdpa"
            except: pass
            print(f"✓ Loaded base FP16 sharded ({dm}).")
            return m
        except Exception as e:
            print(f"[FP16 {dm} failed] Err:", type(e).__name__, str(e)[:200])

    # 3) CPU-only (chậm nhưng luôn chạy được)
    m = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32, device_map={"": "cpu"}).eval()
    print("✓ Loaded base CPU-only.")
    return m

base = load_base()

# ---- Gắn adapter (tránh re-dispatch trên mô hình 4-bit) ----
_prev_map = getattr(base, "hf_device_map", None)
try:
    delattr(base, "hf_device_map")   # chặn Accelerate/PEFT re-dispatch → tránh lỗi bnb uint8
except Exception:
    base.hf_device_map = None

gen = PeftModel.from_pretrained(base, ADAPTER, is_trainable=False).eval()
print("✅ Adapter attached.")

# (khôi phục lại map nếu muốn xem)
if _prev_map is not None:
    base.hf_device_map = _prev_map


# ===== Prompt helpers =====
SYSTEM = (
    "당신은 TOPIK I(1-4급) 객관식 문제 생성기입니다. "
    "오직 하나의 JSON만 출력하세요. "
    "FROM_PASSAGE: ['prompt_ko','choices','answer'] "
    "FULL: ['passage_ko','prompt_ko','choices','answer']."
)
def build_prompt(task, passage=None, hint="일상 대화"):
    if task == "FROM_PASSAGE":
        user = f'TASK: FROM_PASSAGE\n입력지문: "{passage}"\n오직 JSON만.'
    else:
        user = f'TASK: FULL\n주제 힌트: {hint}\n오직 JSON만.'
    msgs=[{"role":"system","content":SYSTEM},{"role":"user","content":user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

# ===== JSON extraction =====
def extract_json_balanced(text: str):
    s = text.find("{")
    if s < 0: return None
    depth = 0; ins = False; esc = False
    for i, ch in enumerate(text[s:], s):
        if ins:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': ins = False
        else:
            if ch == '"': ins = True
            elif ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0: return text[s:i+1]
    return None

# ===== Canonicalization & validation =====
LETTER_SET = {"A", "B", "C", "D"}
NUM2LET = {"1":"A","2":"B","3":"C","4":"D", 1:"A",2:"B",3:"C",4:"D"}

def _to_letter_key(k):
    s = str(k).strip().upper()
    if s in LETTER_SET: return s
    if s in {"1","2","3","4"}: return NUM2LET[s]
    return None

def _normalize_answer(ans):
    if ans is None: return None
    s = str(ans).strip().upper()
    m = re.search(r"[ABCD1-4]", s)
    if not m: return None
    x = m.group(0)
    return x if x in LETTER_SET else NUM2LET.get(x)

def normalize_item(item: dict, *, allow_passage: bool = False):
    """
    Trả về (item_norm, reasons).
    Chuẩn hoá keys -> {"prompt_ko","choices":{"A".."D"},"answer":"A|B|C|D"}.
    Cho phép input có keys 1/2/3/4 hoặc a/b/c/d; có thể giữ 'passage_ko' nếu allow_passage=True.
    """
    reasons = []
    if not isinstance(item, dict):
        return None, ["not a dict"]

    out = {}

    # prompt_ko
    prompt = item.get("prompt_ko")
    if not isinstance(prompt, str) or not prompt.strip():
        prompt = "물음에 맞는 대답을 고르십시오."
        reasons.append("missing/empty prompt_ko -> defaulted")
    out["prompt_ko"] = prompt.strip()

    # choices
    choices = item.get("choices")
    if isinstance(choices, dict):
        canon = {}
        for k, v in choices.items():
            lk = _to_letter_key(k)
            if lk is None:
                reasons.append(f"choices key '{k}' ignored")
                continue
            val = "" if v is None else str(v).strip()
            if not val:
                reasons.append(f"choice {lk} empty")
            canon[lk] = val

        # Nếu chưa đủ A-D, thử rebuild khi keys là số 1..4
        if set(canon.keys()) != LETTER_SET:
            if all(str(k).strip() in {"1","2","3","4"} for k in choices.keys()):
                ordered = [choices.get(i, choices.get(str(i), "")) for i in [1,2,3,4]]
                canon = {l: ("" if v is None else str(v).strip()) for l, v in zip(["A","B","C","D"], ordered)}
            else:
                reasons.append("choices not full A,B,C,D")
        out["choices"] = {k: canon.get(k, "") for k in ["A","B","C","D"]}

    elif isinstance(choices, (list, tuple)) and len(choices) == 4:
        out["choices"] = {l: ("" if v is None else str(v).strip()) for l, v in zip(["A","B","C","D"], choices)}
        reasons.append("choices list -> converted to A-D")
    else:
        return None, ["choices missing or malformed"]

    # answer
    ans = _normalize_answer(item.get("answer"))
    if ans is None:
        reasons.append("answer missing/invalid")
    out["answer"] = ans

    # passage_ko chỉ giữ cho task FULL
    if allow_passage and isinstance(item.get("passage_ko"), str) and item["passage_ko"].strip():
        out["passage_ko"] = item["passage_ko"].strip()

    return out, reasons

def validate_item(item: dict):
    """
    Validate nghiêm sau khi normalize.
    """
    reasons = []
    if not isinstance(item, dict):
        return False, ["not a dict"]

    need = {"prompt_ko", "choices", "answer"}
    if not need.issubset(item.keys()):
        reasons.append("missing prompt_ko/choices/answer")

    ch = item.get("choices", {})
    if not isinstance(ch, dict) or set(ch.keys()) != LETTER_SET:
        reasons.append("choices must have A,B,C,D")

    if item.get("answer") not in LETTER_SET:
        reasons.append("answer not in {A,B,C,D}")

    # chất lượng nhẹ
    if isinstance(ch, dict) and set(ch.keys()) == LETTER_SET:
        vals = [str(ch[k]).strip() for k in ["A","B","C","D"]]
        if any(not v for v in vals):
            reasons.append("empty choice text")
        L = [len(v) for v in vals]
        if L and (max(L) - min(L) > 30):
            reasons.append("choice length variance > 30 chars")

    ok = (len(reasons) == 0)
    return ok, reasons

# ===== Generate (có normalize + validate) =====
def generate_once(task, passage=None, hint="날씨와 약속", sample=False):
    prompt = build_prompt(task, passage=passage, hint=hint)
    ids = tok(prompt, return_tensors="pt").to(gen.device)
    out = gen.generate(
        **ids,
        generation_config=GenerationConfig(
            do_sample=bool(sample),
            temperature=0.0 if not sample else None,
            top_p=1.0,
            max_new_tokens=200,
            pad_token_id=tok.pad_token_id,
        )
    )
    txt = tok.decode(out[0], skip_special_tokens=True)
    js  = extract_json_balanced(txt)
    try:
        parsed = json.loads(js) if js else {}
    except Exception:
        parsed = {}

    allow_passage = (task == "FULL")
    norm, reasons_norm = normalize_item(parsed, allow_passage=allow_passage)
    if norm is None:
        return None, False, ["normalize failed"] + (reasons_norm or []), txt

    ok, reasons_val = validate_item(norm)
    reasons = reasons_norm + ([] if ok else reasons_val)
    return norm, ok, reasons, txt

# ===== Smoke test =====
full_item, full_ok, full_reasons, full_raw = generate_once("FULL", hint="식사와 약속", sample=False)
print("FULL OK:", full_ok, "| reasons:", full_reasons)
print(json.dumps(full_item, ensure_ascii=False, indent=2))

passage = "여자: 주말에 뭐 해요?\n남자: 친구를 만나고 영화 볼 거예요."
fp_item, fp_ok, fp_reasons, fp_raw = generate_once("FROM_PASSAGE", passage=passage, sample=False)
print("\nFROM_PASSAGE OK:", fp_ok, "| reasons:", fp_reasons)
print(json.dumps(fp_item, ensure_ascii=False, indent=2))
