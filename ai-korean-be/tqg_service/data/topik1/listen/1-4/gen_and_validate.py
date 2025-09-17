# gen_and_validate.py
import json, io, os, re, random
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

GEN_CKPT = "ckpts/topik_gen/best"     # ckpt sinh (SFT)
SOLVER_CKPT = "ckpts/topik_mcq/best"  # ckpt chọn đáp án (từ bước baseline)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
JSON_TAG = "@@JSON@@"

# ---------- Load models ----------
gen_tok = AutoTokenizer.from_pretrained(GEN_CKPT, use_fast=True)
if gen_tok.pad_token is None: gen_tok.pad_token = gen_tok.eos_token
gen_model = AutoModelForCausalLM.from_pretrained(GEN_CKPT, torch_dtype=torch.bfloat16 if DEVICE=="cuda" else None).to(DEVICE)

sol_tok = AutoTokenizer.from_pretrained(SOLVER_CKPT, use_fast=True)
if sol_tok.pad_token is None: sol_tok.pad_token = sol_tok.eos_token
solver = AutoModelForCausalLM.from_pretrained(SOLVER_CKPT, torch_dtype=torch.bfloat16 if DEVICE=="cuda" else None).to(DEVICE)

# ---------- Prompt builders ----------
def build_prompt_full(controls: Dict[str, Any], seed: Dict[str, Any]) -> str:
    ctrl = (f"<SECTION={controls['section']}> <LEVEL={controls['level']}> "
            f"<MODE={controls['mode']}> <NUMQ={controls['num_questions']}> "
            f"<LEN={controls['length_passage_chars'][0]}-{controls['length_passage_chars'][1]}> "
            f"<ONLYONE={int(controls['only_one_correct'])}>")
    return (
        f"{ctrl}\n"
        "TOPIK 스타일로 1개의 지문(듣기용)과 4지선다형 문항 1개를 JSON으로 생성하세요.\n"
        "- 지문/대화는 TOPIK I 어휘, 짧고 자연스럽게.\n"
        "- 보기는 A/B/C/D, 오직 하나만 정답.\n"
        "- JSON 키: passage_ko, questions[{prompt_ko, choices[{label,text_ko}], answer, rationale_ko}]\n"
        f"SEED: {json.dumps(seed, ensure_ascii=False)}\n\n"
        f"{JSON_TAG}\n"
    )

# ---------- Generation / JSON parse ----------
def extract_json(text: str) -> dict:
    # lấy phần sau TAG, cắt tới dấu ngoặc khớp
    if JSON_TAG in text:
        text = text.split(JSON_TAG, 1)[1]
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m: raise ValueError("no_json_block")
    block = m.group(0)
    # sửa lỗi dấu phẩy thừa nhỏ (có thể mở rộng nếu cần)
    block = re.sub(r",\s*([}\]])", r"\1", block)
    return json.loads(block)

def generate_item(controls: Dict[str,Any], seed: Dict[str,Any], max_new_tokens=512, temperature=0.2, top_p=0.9) -> dict:
    prompt = build_prompt_full(controls, seed)
    inpt = gen_tok(prompt, return_tensors="pt").to(DEVICE)
    out = gen_model.generate(
        **inpt, max_new_tokens=max_new_tokens, do_sample=True,
        temperature=temperature, top_p=top_p, eos_token_id=gen_tok.eos_token_id
    )
    gen_txt = gen_tok.decode(out[0], skip_special_tokens=True)
    obj = extract_json(gen_txt)
    return obj

# ---------- Validation ----------
def validate_schema(obj: dict) -> bool:
    if "passage_ko" not in obj or "questions" not in obj: return False
    qs = obj["questions"]
    if not isinstance(qs, list) or not qs: return False
    q = qs[0]
    if "prompt_ko" not in q or "choices" not in q or "answer" not in q: return False
    ch = q["choices"]
    if not (isinstance(ch, list) and len(ch) == 4): return False
    labs = [c.get("label") for c in ch]
    if labs != ["A","B","C","D"]: return False
    if q["answer"] not in {"A","B","C","D"}: return False
    # nội dung rỗng?
    if any(not (c.get("text_ko") or "").strip() for c in ch): return False
    # trùng lựa chọn
    texts = [c["text_ko"].strip() for c in ch]
    if len(set(texts)) < 4: return False
    # passage ngắn hợp lệ
    if len(obj["passage_ko"]) < 6: return False
    return True

def solver_predict(passage: str, choices: List[str], prompt_ko: str) -> str:
    instr = f"{prompt_ko}\n\n지문:\n{passage}\n\n보기:\n" + "\n".join(choices) + "\n\n정답은 A/B/C/D 중 하나의 글자만 쓰세요.\n@@ANSWER@@ "
    inpt = sol_tok(instr, return_tensors="pt").to(DEVICE)
    out = solver.generate(**inpt, max_new_tokens=1, do_sample=False)
    tail = sol_tok.decode(out[0][inpt["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    m = re.match(r"^[ABCD]", tail)
    return m.group(0) if m else "A"

def validate_with_solver(obj: dict) -> bool:
    q = obj["questions"][0]
    passage = obj["passage_ko"].strip()
    prompt = q["prompt_ko"].strip()
    choices = [f"{c['label']}) {c['text_ko']}" for c in q["choices"]]
    pred = solver_predict(passage, choices, prompt)
    return pred == q["answer"]

# ---------- Main loop ----------
def main():
    os.makedirs("generated/topik", exist_ok=True)
    controls = {"section":"Listening","level":1,"mode":"utterance","num_questions":1,"length_passage_chars":[40,90],"only_one_correct":True}

    seeds = [
        {"topic":"weather","key_phrases_ko":["날씨","상태"],"structure":["시작","핵심정보","마무리"]},
        {"topic":"quantity","key_phrases_ko":["수량","사과"],"structure":["질문","대답"]},
        {"topic":"identity","key_phrases_ko":["정체성","물건"],"structure":["질문","대답"]},
        {"topic":"method","key_phrases_ko":["방법","교통수단"],"structure":["질문","대답"]},
    ]

    accepted = []
    for i in range(200):  # sinh 200 mẫu thử
        sd = random.choice(seeds)
        try:
            obj = generate_item(controls, sd, temperature=0.25, top_p=0.9)
            if not validate_schema(obj): continue
            if not validate_with_solver(obj): continue
            accepted.append(obj)
        except Exception:
            continue

    # lưu gói sinh ra (pack 1 câu/hàng)
    with io.open("generated/topik/pq_gen_full.accepted.jsonl","w",encoding="utf-8") as g:
        for o in accepted:
            g.write(json.dumps(o, ensure_ascii=False)+"\n")
    print("accepted:", len(accepted))

if __name__ == "__main__":
    main()
