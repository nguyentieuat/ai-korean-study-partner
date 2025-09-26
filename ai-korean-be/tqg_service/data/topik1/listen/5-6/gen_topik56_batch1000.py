# %%writefile gen_topik56_batch1000.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, re, json, itertools, random
from collections import Counter
from typing import Tuple, Any, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ================== CONFIG ==================
BASE_MODEL  = "meta-llama/Meta-Llama-3-8B-Instruct"
ADAPTER_DIR = "nguyentieuat/topik-56-llama3-8b-lora"  # HF repo LoRA
USE_4BIT    = True   # cần GPU, BnB 4-bit

# Tối ưu tốc độ/độ ổn định JSON
MAX_NEW_TOKENS     = 128         # 80~120 là hợp lý cho JSON ngắn
TEMPERATURE        = 0.6
TOP_P              = 0.9
REPETITION_PENALTY = 1.08

NUM_TOTAL          = 1000
SAVE_PATH          = "/kaggle/working/topik56_batch1000.jsonl"  # đổi nếu cần
BATCH_SIZE         = 12         # 8–16 cho T4; A100 có thể 16–24
CHECKPOINT_EVERY   = 100         # ghi sớm để thấy tiến độ
MAX_ATTEMPTS       = 30000      # lượt batch thử tối đa

TOPICS = (
    "shopping","restaurant","transport","school","work",
    "home","bank_post","social","travel_hotel","weather"
)

# ================== PROMPT & VALIDATOR ==================
EXPECTED_TITLE = "[5~6] 다음을 듣고 <보기>와 같이 이어지는 말을 고르십시오."
RESPONSE_TAG   = "#### JSON_START ####"

INSTR = """### Instruction:
TOPIK I (초급) 5~6 유형 문제를 '처음부터' 생성하세요. 출력은 JSON 한 개만.
필수 스키마(값 고정/규칙):
- "type": "Nghe_Loi_Tiep"
- "section": "Nghe"
- "level": "1"
- "title": "[5~6] 다음을 듣고 <보기>와 같이 이어지는 말을 고르십시오."
- "question": "이어지는 말을 고르십시오."
- "explanation": ""
- score 규칙: question_no=5 이면 4, question_no=6 이면 3.
- options: 정확히 A,B,C,D (모두 서로 달라야 함, 각 30자 이하, 한글 포함),
- answer: A|B|C|D (options 키 중 하나),
- dialogue: 길이 1, 형식: [ { "speaker": 남자|여자, "text": (한글, 40자 이하) } ],
- question_no: 5|6,
주의:
- JSON 외의 텍스트 출력 금지 (JSON은 '{'로 시작해 '}'로 끝남).
"""

PROMPT_TMPL = (
    "{instr}\n\n"
    "### Controls:\n"
    "topic: {topic}\n"
    "question_no: {qno}\n"
    "speaker: {speaker}  # 반드시 이 화자로 작성\n\n"
    f"{RESPONSE_TAG}\n"
    "{scaffold}"
)

def scaffold_head(qno:int) -> str:
    score = 4 if qno == 5 else 3
    return (
        '{\n'
        '  "type": "Nghe_Loi_Tiep",\n'
        '  "section": "Nghe",\n'
        '  "level": "1",\n'
        f'  "title": "{EXPECTED_TITLE}",\n'
        '  "question": "이어지는 말을 고르십시오.",\n'
        '  "explanation": "",\n'
        f'  "score": {score},\n'
    )

def build_prompt(topic: str, qno: int, speaker: str) -> str:
    return PROMPT_TMPL.format(
        instr=INSTR, topic=topic, qno=qno, speaker=speaker, scaffold=scaffold_head(qno)
    )

NEED_KEYS = {"type","section","level","title","question","explanation",
             "score","options","answer","dialogue","question_no"}

def validate_json(s: str) -> Tuple[bool, Any]:
    try:
        obj = json.loads(s)
    except Exception:
        return False, "parse"

    miss = NEED_KEYS - obj.keys()
    if miss: return False, f"missing:{sorted(list(miss))}"

    if obj.get("type") != "Nghe_Loi_Tiep": return False, "type_value"
    if obj.get("section") != "Nghe": return False, "section_value"
    if obj.get("level") != "1": return False, "level_value"
    if obj.get("title") != EXPECTED_TITLE: return False, "title_value"
    if obj.get("question") != "이어지는 말을 고르십시오.": return False, "question_value"
    if obj.get("explanation") != "": return False, "explanation_value"

    if obj.get("answer") not in {"A","B","C","D"}: return False, "answer_label"

    opts = obj.get("options")
    if not isinstance(opts, dict): return False, "options_not_dict"
    if set(opts.keys()) != {"A","B","C","D"}: return False, "options_keys"

    dlg = obj.get("dialogue")
    if not isinstance(dlg, list) or len(dlg) != 1: return False, "dialogue_len"
    spk = dlg[0].get("speaker")
    if spk not in ("남자","여자"): return False, "speaker"

    if obj.get("question_no") not in (5,6): return False, "question_no"

    def is_ko(x: str) -> bool:
        return bool(re.search(r"[\uac00-\ud7af]", x or ""))

    dtext = str(dlg[0].get("text","")).strip()
    if not is_ko(dtext): return False, "dialogue_not_korean"
    if len(dtext) > 40: return False, "dialogue_too_long"

    vals = [str(v).strip() for v in opts.values()]
    if any(len(v)==0 for v in vals): return False, "empty_option"
    if len(set(vals)) < 4: return False, "duplicate_options"
    for v in vals:
        if not is_ko(v): return False, "option_not_korean"
        if len(v) > 30: return False, "option_too_long"

    return True, obj

def normalize_item(obj):
    """Khoá các trường cố định + score theo qno (5->4, 6->3)."""
    obj["type"] = "Nghe_Loi_Tiep"
    obj["section"] = "Nghe"
    obj["level"] = "1"
    obj["title"] = EXPECTED_TITLE
    obj["question"] = "이어지는 말을 고르십시오."
    obj["explanation"] = ""
    qno = int(obj.get("question_no", 5))
    obj["score"] = 4 if qno == 5 else 3
    return obj

# ================== MODEL ==================
def load_model():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"       # <— quan trọng cho decoder-only
    tok.truncation_side = "left" 

    if USE_4BIT:
        if not torch.cuda.is_available():
            raise RuntimeError("BitsAndBytes 4-bit cần CUDA. Hãy bật GPU runtime.")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=bnb, device_map="auto"
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float32, device_map={"": "cpu"}
        )
    model = PeftModel.from_pretrained(base, ADAPTER_DIR, device_map="auto")
    try:
        model.eval()
        model.config.use_cache = False
    except Exception:
        pass
    return tok, model

# ================== GENERATION ==================
def _cut_after_tag(text: str) -> str:
    body = text.split(RESPONSE_TAG)[-1].strip()
    if body.count("{")>0 and body.count("}")>0:
        end = body.rfind("}")
        if end!=-1: body = body[:end+1]
    return body

def generate_batch(tok, model, topics: List[str], qno: int, speaker: str) -> List[str]:
    # 1) Tính input_max_len = (context window) - (budget sinh) - margin
    ctx = getattr(getattr(model, "config", None), "max_position_embeddings", 8192)
    input_max_len = max(256, ctx - MAX_NEW_TOKENS - 16)  # chừa chỗ cho output + margin

    prompts = [build_prompt(t, qno, speaker) for t in topics]
    enc = tok(
        prompts,
        return_tensors="pt",
        padding=True,           # sẽ là left-padding nhờ tok.padding_side="left"
        truncation=True,        # tránh dài quá context
        max_length=input_max_len,
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
    decoded = tok.batch_decode(out, skip_special_tokens=True)
    return [_cut_after_tag(x) for x in decoded]

# ================== DRIVER (balanced 1000) ==================
def run_balanced_1000(save_path=SAVE_PATH,
                      topics=TOPICS,
                      batch_size=BATCH_SIZE,
                      checkpoint_every=CHECKPOINT_EVERY,
                      max_attempts=MAX_ATTEMPTS):
    slots = [(5,"남자"), (5,"여자"), (6,"남자"), (6,"여자")]
    target = {s: NUM_TOTAL // 4 for s in slots}
    # đảm bảo đúng 1000 (trường hợp NUM_TOTAL không chia hết)
    leftovers = NUM_TOTAL - sum(target.values())
    for i in range(leftovers):
        target[slots[i % 4]] += 1

    got = Counter()
    rows, buffer = [], []

    # reset file
    try:
        if os.path.exists(save_path):
            os.remove(save_path)
    except Exception:
        pass

    def _flush_buffer():
        nonlocal buffer
        if not buffer: return
        tmp_path = save_path + ".part"
        with open(tmp_path, "a", encoding="utf-8") as f:
            for line in buffer: f.write(line + "\n")
            f.flush(); os.fsync(f.fileno())
        with open(tmp_path, "r", encoding="utf-8") as fin, open(save_path, "a", encoding="utf-8") as fout:
            for line in fin: fout.write(line)
            fout.flush(); os.fsync(fout.fileno())
        try: os.remove(tmp_path)
        except Exception: pass
        buffer = []

    topic_cycle = itertools.cycle(topics)
    attempts = 0

    # model+tokenizer
    print("[INFO] Loading model + LoRA …")
    tok, model = load_model()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    while sum(got.values()) < NUM_TOTAL and attempts < max_attempts:
        # Chọn ô thiếu nhất
        need_sorted = sorted(slots, key=lambda s: (target[s] - got[s]), reverse=True)
        qno, want_spk = next((q, sp) for (q, sp) in need_sorted if got[(q, sp)] < target[(q, sp)])

        # Chuẩn bị topics cho batch
        bs = min(batch_size, target[(qno, want_spk)] - got[(qno, want_spk)])
        batch_topics = [next(topic_cycle) for _ in range(max(bs, 1))]

        bodies = generate_batch(tok, model, batch_topics, qno=qno, speaker=want_spk)
        attempts += 1

        for body in bodies:
            if sum(got.values()) >= NUM_TOTAL:
                break
            ok, obj = validate_json(body)
            if not ok:
                continue
            # speaker kiểm tra lại
            spk = obj["dialogue"][0].get("speaker")
            if spk != want_spk:
                continue

            fixed = normalize_item(obj)
            rows.append(fixed)
            buffer.append(json.dumps(fixed, ensure_ascii=False))
            got[(qno, spk)] += 1

            total_now = sum(got.values())
            if total_now % 50 == 0:
                print(f"[PROG] {total_now}/{NUM_TOTAL} "
                      f"(5남:{got[(5,'남자')]}/{target[(5,'남자')]}, "
                      f"5여:{got[(5,'여자')]}/{target[(5,'여자')]}, "
                      f"6남:{got[(6,'남자')]}/{target[(6,'남자')]}, "
                      f"6여:{got[(6,'여자')]}/{target[(6,'여자')]})")

            if len(buffer) >= checkpoint_every:
                _flush_buffer()

    _flush_buffer()
    print(f"[SAVE] {len(rows)} items -> {save_path}")
    print("[TARGET]", target)
    print("[GOT]", dict(got))
    if len(rows) < NUM_TOTAL:
        print("[WARN] thiếu:", {k: target[k]-got[k] for k in target if got[k] < target[k]})

    dist_ans = Counter(r["answer"] for r in rows)
    dist_qno = Counter(r["question_no"] for r in rows)
    dist_spk = Counter(r["dialogue"][0]["speaker"] for r in rows)
    print("[DIST] answer:", dict(dist_ans))
    print("[DIST] qno:", dict(dist_qno))
    print("[DIST] speaker:", dict(dist_spk))
    return save_path

# ================== MAIN ==================
if __name__ == "__main__":
    # Smoke 1 mẫu (tùy chọn)
    # tok, model = load_model()
    # body = generate_batch(tok, model, ["shopping"], qno=5, speaker="남자")[0]
    # print(validate_json(body))

    # Chạy balanced 1000
    out = run_balanced_1000(
        save_path=SAVE_PATH,
        topics=TOPICS,
        batch_size=BATCH_SIZE,
        checkpoint_every=CHECKPOINT_EVERY,
        max_attempts=MAX_ATTEMPTS,
    )
    print("DONE:", out)
