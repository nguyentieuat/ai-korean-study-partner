# ================= TOPIK MCQ Synthesizer (GEN + MCP, no answer balancing) =================
import os, re, json, time, random, hashlib, gc
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, GenerationConfig,
    LogitsProcessorList, InfNanRemoveLogitsProcessor,
    RepetitionPenaltyLogitsProcessor, TopPLogitsWarper, TemperatureLogitsWarper,
)
from peft import PeftModel
from huggingface_hub import login

# ---------------- Config ----------------
BASE         = "meta-llama/Meta-Llama-3-8B-Instruct"
ADAPTER_GEN  = "nguyentieuat/topik-gen-llama3-8b-lora"   # adapter sinh đề
ADAPTER_MCP  = "nguyentieuat/topik-mcq-llama3-8b-lora"   # adapter chọn đáp án
OUT_PATH     = "./topik_mcq_1000.jsonl"                  # file output
N_SAMPLES    = 1000                                      # số câu cần sinh

# (nếu base gated) đăng nhập HF
try: login()
except Exception: pass

# Giảm phân mảnh VRAM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True
random.seed(13); torch.manual_seed(13)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(13)

# ---------------- Tokenizer ----------------
tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

# ---------------- Load base (4-bit → FP16 → CPU) ----------------
def load_base_model():
    # 1) 4-bit
    try:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        m = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto").eval()
        print("✓ Loaded base in 4-bit.")
        return m
    except Exception as e:
        print("[4-bit failed] → fallback FP16 sharded. Err:", type(e).__name__, str(e)[:220])

    # 2) FP16 sharded
    max_mem = {i: "7GiB" for i in range(torch.cuda.device_count())}
    max_mem["cpu"] = "32GiB"
    os.makedirs("./offload", exist_ok=True)
    for dm in ("balanced_low_0","auto"):
        try:
            m = AutoModelForCausalLM.from_pretrained(
                BASE, torch_dtype=torch.float16, device_map=dm,
                max_memory=max_mem, low_cpu_mem_usage=True, offload_folder="./offload"
            ).eval()
            try: m.config.attn_implementation = "sdpa"
            except: pass
            print(f"✓ Loaded base FP16 sharded ({dm}).")
            return m
        except Exception as e:
            print(f"[FP16 {dm} failed] Err:", type(e).__name__, str(e)[:220])

    # 3) CPU only
    m = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32, device_map={"": "cpu"}).eval()
    print("✓ Loaded base CPU-only.")
    return m

base = load_base_model()

# ---------------- Attach adapters (patch tránh re-dispatch 4-bit) ----------------
def attach_gen_adapter(_base, repo_id, name_hint="gen"):
    prev = getattr(_base, "hf_device_map", None)
    try: delattr(_base, "hf_device_map")
    except Exception: _base.hf_device_map = None
    peft = PeftModel.from_pretrained(_base, repo_id, is_trainable=False).eval()
    print("✅ GEN adapter attached.")
    if prev is not None:
        _base.hf_device_map = prev
    return peft

def attach_extra_adapter(_peft, repo_id, adapter_name):
    host = getattr(_peft, "base_model", _peft)
    prev = getattr(host, "hf_device_map", None)
    try: delattr(host, "hf_device_map")
    except Exception: host.hf_device_map = None
    _peft.load_adapter(repo_id, adapter_name=adapter_name)
    print(f"✅ Extra adapter attached: {adapter_name}")
    if prev is not None:
        host.hf_device_map = prev

gen = attach_gen_adapter(base, ADAPTER_GEN)  # PeftModel
attach_extra_adapter(gen, ADAPTER_MCP, adapter_name="mcp")
try: gen.set_adapter("default")  # quay lại GEN sau khi load
except Exception: pass

# ---------------- Helpers: JSON, chuẩn hoá, validate ----------------
LETTER_SET = {"A","B","C","D"}
NUM2LET = {"1":"A","2":"B","3":"C","4":"D",1:"A",2:"B",3:"C",4:"D"}

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

def normalize_item(item: dict, *, allow_passage=False):
    reasons=[]
    if not isinstance(item, dict): return None, ["not a dict"]
    out={}
    # prompt
    prompt=item.get("prompt_ko")
    if not isinstance(prompt, str) or not prompt.strip():
        prompt="문맥에 맞는 대답을 고르십시오."; reasons.append("default prompt_ko")
    out["prompt_ko"]=prompt.strip()
    # choices
    ch=item.get("choices")
    if isinstance(ch, dict):
        canon={}
        for k,v in ch.items():
            lk=_to_letter_key(k)
            if lk is None: continue
            canon[lk]=("" if v is None else str(v).strip())
        if set(canon.keys())!=LETTER_SET:
            if all(str(k).strip() in {"1","2","3","4"} for k in ch.keys()):
                ordered=[ch.get(i, ch.get(str(i), "")) for i in [1,2,3,4]]
                canon={L:("" if v is None else str(v).strip()) for L,v in zip(["A","B","C","D"], ordered)}
        if set(canon.keys())!=LETTER_SET:
            return None, ["choices not full A-D"]
        out["choices"]={k:canon[k] for k in ["A","B","C","D"]}
    elif isinstance(ch,(list,tuple)) and len(ch)==4:
        out["choices"]={L:(str(v).strip() if v else "") for L,v in zip(["A","B","C","D"], ch)}
    else:
        return None, ["choices malformed"]
    # answer
    ans=_normalize_answer(item.get("answer"))
    out["answer"]=ans
    # passage
    if allow_passage and isinstance(item.get("passage_ko"),str) and item["passage_ko"].strip():
        out["passage_ko"]=item["passage_ko"].strip()
    return out, reasons

def validate_item(item: dict):
    reasons=[]
    if not isinstance(item, dict): return False, ["not a dict"]
    if not {"prompt_ko","choices","answer"}.issubset(item.keys()):
        reasons.append("missing prompt_ko/choices/answer")
    ch=item.get("choices",{})
    if not isinstance(ch, dict) or set(ch.keys())!=LETTER_SET:
        reasons.append("choices must have A,B,C,D")
    if item.get("answer") not in LETTER_SET:
        reasons.append("answer not in {A,B,C,D}")
    if isinstance(ch, dict) and set(ch.keys())==LETTER_SET:
        vals=[str(ch[k]).strip() for k in ["A","B","C","D"]]
        if any(not v for v in vals): reasons.append("empty choice")
        L=[len(v) for v in vals]
        if L and (max(L)-min(L)>30): reasons.append("choice length variance > 30")
    return len(reasons)==0, reasons

# ---------------- Prompting ----------------
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

def generate_json(task, passage=None, hint="일상 대화", sample=True):
    prompt = build_prompt(task, passage=passage, hint=hint)
    ids = tok(prompt, return_tensors="pt").to(gen.device)
    if sample:
        processors = LogitsProcessorList([
            InfNanRemoveLogitsProcessor(),
            RepetitionPenaltyLogitsProcessor(1.12),
            TemperatureLogitsWarper(0.8),
            TopPLogitsWarper(0.9),
        ])
        out = gen.generate(**ids, do_sample=True, logits_processor=processors, max_new_tokens=200, use_cache=True)
    else:
        out = gen.generate(**ids, do_sample=False, max_new_tokens=200, use_cache=True)
    txt = tok.decode(out[0], skip_special_tokens=True)
    js  = extract_json_balanced(txt)
    try:
        data = json.loads(js) if js else {}
    except Exception:
        data = {}
    norm, _ = normalize_item(data, allow_passage=(task=="FULL"))
    return norm

# ---------------- FULL robust ----------------
def generate_passage(hint="일상 대화", sample=True):
    sys = ("당신은 TOPIK I용 짧은 지문/대화를 만듭니다. "
           "한국어 한 단락(1~2문장)만 출력. 다른 텍스트/따옴표/JSON 금지.")
    user = f"주제 힌트: {hint}\n형식 예: 남자: … 여자: … 또는 1~2문장 서술"
    msgs=[{"role":"system","content":sys},{"role":"user","content":user}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(gen.device)
    out = gen.generate(**ids, do_sample=sample, max_new_tokens=80, use_cache=True,
                       temperature=0.8 if sample else None, top_p=0.9 if sample else None,
                       repetition_penalty=1.1)
    txt = tok.decode(out[0], skip_special_tokens=True).strip()
    txt = re.sub(r"^```.*?\n|\n```$", "", txt, flags=re.DOTALL).strip()
    txt = re.sub(r'^[\'"“”]+|[\'"“”]+$', '', txt).strip()
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    if len(lines)>2: lines=lines[:2]
    return " ".join(lines)

def generate_full_robust(hint="일상 대화", sample=True):
    it = generate_json("FULL", hint=hint, sample=sample)
    if it and it.get("passage_ko"):
        return it
    p = generate_passage(hint=hint, sample=sample)
    it2 = generate_json("FROM_PASSAGE", passage=p, sample=sample)
    if it2:
        it2["passage_ko"]=p
    return it2

# ---------------- MCP (answer selector) ----------------
def build_clf_prompt(item):
    passage = item.get("passage_ko","").strip()
    prompt_ko = item.get("prompt_ko","문맥에 맞는 대답을 고르십시오.").strip()
    ch = item["choices"]
    lines=[]
    if passage: lines.append(f"지문:\n{passage}\n")
    lines.append(f"문제: {prompt_ko}")
    lines.append(f"보기:\nA) {ch['A']}\nB) {ch['B']}\nC) {ch['C']}\nD) {ch['D']}")
    lines.append("정답만 A/B/C/D 중 하나로 출력하세요.")
    return "\n".join(lines)

def mcp_predict(item):
    # switch sang adapter MCP
    try: gen.set_adapter("mcp")
    except Exception: pass
    msgs=[{"role":"system","content":"너는 객관식 정답 선택기이다. 반드시 A/B/C/D 중 하나만 출력해라."},
          {"role":"user","content": build_clf_prompt(item)}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(gen.device)
    out = gen.generate(**ids, do_sample=False, max_new_tokens=2)
    pred = tok.decode(out[0], skip_special_tokens=True)
    m = re.search(r"[ABCD]", pred)
    # quay lại adapter GEN
    try: gen.set_adapter("default")
    except Exception: pass
    return m.group(0) if m else None

# ---------------- Dedup + Synthesizer (no balancing) ----------------
def item_sig(it: dict):
    p = it.get("passage_ko","").strip()
    ch = it.get("choices",{})
    key = "|".join([p, it.get("prompt_ko","").strip()] + [ch.get(k,"").strip() for k in ("A","B","C","D")])
    return hashlib.md5(key.encode("utf-8")).hexdigest()

def synthesize_dataset(n=N_SAMPLES, save_path=OUT_PATH,
                       hints=None, sample=True, per_checkpoint=100, max_tries=None):
    if max_tries is None: max_tries = n * 10
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    items, seen = [], set()
    counts = {"A":0,"B":0,"C":0,"D":0}
    i_hint, tries = 0, 0

    while len(items) < n and tries < max_tries:
        tries += 1
        hint = (hints[i_hint % len(hints)] if hints else "일상 대화")
        i_hint += 1

        cand = generate_full_robust(hint=hint, sample=sample)
        if not cand: 
            continue
        ok, _ = validate_item(cand)
        if not ok:
            continue

        pred = mcp_predict(cand)
        if pred is None or pred != cand["answer"]:
            continue

        sig = item_sig(cand)
        if sig in seen:
            continue

        items.append(cand)
        seen.add(sig)
        counts[cand["answer"]] += 1

        if len(items) % per_checkpoint == 0:
            with open(save_path, "w", encoding="utf-8") as f:
                for ex in items:
                    f.write(json.dumps(ex, ensure_ascii=False)+"\n")
            print(f"[{len(items)}/{n}] checkpoint saved → {save_path} | counts={counts}")
        time.sleep(0.02)

    with open(save_path, "w", encoding="utf-8") as f:
        for ex in items:
            f.write(json.dumps(ex, ensure_ascii=False)+"\n")
    print(f"Done. Saved {len(items)} items to {save_path} | counts={counts}")
    return items, counts

# ---------------- Run ----------------
hint_pool = ["날씨와 약속","식사와 주문","쇼핑과 가격","교통과 길찾기","가족과 취미","학교와 수업","직장과 일정"]

items, counts = synthesize_dataset(
    n=N_SAMPLES,
    save_path=OUT_PATH,
    hints=hint_pool,
    sample=True,            # bật sampling để đa dạng
    per_checkpoint=100      # lưu mỗi 100 mẫu
)

# Xem vài mẫu
print("\n--- Preview 3 items ---")
for ex in items[:3]:
    print(json.dumps(ex, ensure_ascii=False, indent=2))
