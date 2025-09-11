# ==== TOPIK MCQ Answer Selector — Inference (4-bit → FP16 → CPU fallback) ====
import os, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel

# ---- Config (bạn yêu cầu) ----
BASE    = "meta-llama/Meta-Llama-3-8B-Instruct"          # base gated -> cần login HF nếu chưa có quyền
ADAPTER = "nguyentieuat/topik-mcq-llama3-8b-lora"        # adapter trên Hugging Face

# (tuỳ chọn) login HF nếu base model bị gated
try:
    from huggingface_hub import login
    login()  # dán token nếu được hỏi
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

clf = PeftModel.from_pretrained(base, ADAPTER, is_trainable=False).eval()
print("✅ Adapter attached.")

# (khôi phục lại map nếu muốn xem)
if _prev_map is not None:
    base.hf_device_map = _prev_map

# ---- Hàm suy luận: trả về 'A' | 'B' | 'C' | 'D' ----
SYSTEM = "너는 객관식 정답 선택기이다. 반드시 A/B/C/D 중 하나만 출력해라."
GENCFG = GenerationConfig(do_sample=False, max_new_tokens=3)  # greedy để giữ format

def classify_answer(passage_ko: str | None, prompt_ko: str, choices: dict) -> str | None:
    """
    passage_ko: có thể None hoặc "" nếu không có đoạn văn
    prompt_ko : câu hỏi (ví dụ: '문맥에 맞는 대답을 고르십시오.')
    choices   : dict {'A':..., 'B':..., 'C':..., 'D':...}
    """
    # Chuẩn hoá key
    if set(choices.keys()) != {"A","B","C","D"}:
        # thử map từ '1','2','3','4' → 'A'..'D' hoặc viết hoa
        if set(choices.keys()) == {"1","2","3","4"}:
            choices = {L: choices[k] for L,k in zip(["A","B","C","D"], ["1","2","3","4"])}
        else:
            choices = {k.upper(): v for k,v in choices.items()}
    # soạn prompt
    psg = passage_ko.strip() if passage_ko else "(없음)"
    text = (
        f"지문:\n{psg}\n\n문제: {prompt_ko}\n보기:\n"
        f"A) {choices['A']}\nB) {choices['B']}\nC) {choices['C']}\nD) {choices['D']}\n\n"
        "정답만 A/B/C/D 중 하나로 출력하세요."
    )
    msgs = [{"role":"system","content":SYSTEM}, {"role":"user","content":text}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(clf.device)
    out = clf.generate(**ids, generation_config=GENCFG)
    pred = tok.decode(out[0], skip_special_tokens=True)
    m = re.search(r"[ABCD]", pred)
    return m.group(0) if m else None

# ---- Smoke test ----
if __name__ == "__main__":
    passage = "여자: 오늘 날씨가 어때요?"
    prompt  = "문맥에 맞는 대답을 고르십시오."
    choices = {"A":"맑아요.","B":"내일 좋아요.","C":"여름이에요.","D":"오후에 비가 와요."}
    ans = classify_answer(passage, prompt, choices)
    print("Pred:", ans)  # kỳ vọng 'A' hoặc 'D' tuỳ dataset của bạn
