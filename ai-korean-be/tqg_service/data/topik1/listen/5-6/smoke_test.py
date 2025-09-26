# %%writefile smoke_infer_topik56.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SMOKE INFER — TOPIK 5–6 (match TRAIN prompt/marker)
- Load base + LoRA từ HF
- Prompt kết thúc bằng '#### JSON_START ####' (đúng như TRAIN)
- Generate 1 mẫu + validate
"""

import os, json, re, random
from typing import Tuple, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ========= CONFIG =========
BASE_MODEL  = "meta-llama/Meta-Llama-3-8B-Instruct"
ADAPTER_DIR = "nguyentieuat/topik-56-llama3-8b-lora"   # HF repo id của bạn
USE_4BIT    = True

MAX_NEW_TOKENS       = 200
TEMPERATURE          = 0.6
TOP_P                = 0.9
REPETITION_PENALTY   = 1.12

# ========= PROMPT (KHỚP TRAIN) =========
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

RESPONSE_TAG = "#### JSON_START ####"

PROMPT_TMPL = (
    "{instr}\n\n"
    "### Controls:\n"
    "topic: {topic}\n"
    "question_no: {qno}\n\n"
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

def build_prompt(topic: str, qno: int) -> str:
    return PROMPT_TMPL.format(instr=INSTR, topic=topic, qno=qno, scaffold=scaffold_head(qno))

# ========= VALIDATOR =========
NEED_KEYS = {"type","section","level","title","question","explanation","score","options","answer","dialogue","question_no"}

def is_korean_text(s: str) -> bool:
    return bool(re.search(r"[\uac00-\ud7af]", s or ""))

EXPECTED_TITLE = "[5~6] 다음을 듣고 <보기>와 같이 이어지는 말을 고르십시오."



def validate_json(s: str) -> Tuple[bool, Any]:
    try:
        obj = json.loads(s)
    except Exception:
        return False, "parse"

    miss = NEED_KEYS - obj.keys()
    if miss: return False, f"missing:{sorted(list(miss))}"

    # --- NEW: khóa các giá trị bắt buộc ---
    if obj.get("type") != "Nghe_Loi_Tiep": return False, "type_value"
    if obj.get("section") != "Nghe": return False, "section_value"
    if obj.get("level") != "1": return False, "level_value"
    if obj.get("title") != EXPECTED_TITLE: return False, "title_value"
    if obj.get("question") != "이어지는 말을 고르십시오.": return False, "question_value"
    if obj.get("explanation") != "": return False, "explanation_value"

    # options & answer
    if not isinstance(obj.get("options"), dict): return False, "options_not_dict"
    if set(obj["options"].keys()) != {"A","B","C","D"}: return False, "options_keys"
    if obj.get("answer") not in {"A","B","C","D"}: return False, "answer_label"

    # dialogue
    dlg = obj.get("dialogue")
    if not isinstance(dlg, list) or len(dlg) != 1: return False, "dialogue_len"
    spk = dlg[0].get("speaker")
    if spk not in ("남자","여자"): return False, "speaker"

    # question_no
    if obj.get("question_no") not in (5,6): return False, "question_no"

    # nội dung Hàn + giới hạn độ dài
    def is_ko(x): return bool(re.search(r"[\uac00-\ud7af]", x or ""))

    dtext = str(dlg[0].get("text","")).strip()
    if not is_ko(dtext): return False, "dialogue_not_korean"
    if len(dtext) > 40: return False, "dialogue_too_long"

    vals = [str(v).strip() for v in obj["options"].values()]
    if any(len(v)==0 for v in vals): return False, "empty_option"
    if len(set(vals)) < 4: return False, "duplicate_options"
    for v in vals:
        if not is_ko(v): return False, "option_not_korean"
        if len(v) > 30: return False, "option_too_long"

    return True, obj

# ========= MODEL LOADER =========
def load_model():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = None
    if USE_4BIT:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto")
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    try:
        model.eval(); model.config.use_cache = False
    except Exception:
        pass
    return tok, model

# ========= GENERATE =========
def generate_one(topic="shopping", qno=5, seed=None):
    import torch
    if seed is None: seed = random.randint(10**6, 10**7-1)
    torch.manual_seed(seed)
    prompt = build_prompt(topic, qno)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        eos_token_id=tok.eos_token_id,
    )
    full = tok.decode(out[0], skip_special_tokens=True)
    # Cắt theo marker TRAIN
    body = full.split(RESPONSE_TAG)[-1].strip()
    if body.count("{")>0 and body.count("}")>0:
        end = body.rfind("}")
        if end!=-1: body = body[:end+1]
    return body, seed

# ==== Balanced batch generator: 1000 items, balanced by (qno, speaker) ====
from collections import Counter
import os, itertools, json

def normalize_item(obj):
    """Chuẩn hoá trường cố định + score theo qno (5->4, 6->3)."""
    obj["type"] = "Nghe_Loi_Tiep"
    obj["section"] = "Nghe"
    obj["level"] = "1"
    obj["title"] = "[5~6] 다음을 듣고 <보기>와 같이 이어지는 말을 고르십시오."
    obj["question"] = "이어지는 말을 고르십시오."
    obj["explanation"] = ""
    qno = int(obj.get("question_no", 5))
    obj["score"] = 4 if qno == 5 else 3
    return obj

def run_balanced_batch(num_total=1000,
                       topics=("shopping","restaurant","transport","school","work","home","bank_post","social","travel_hotel","weather"),
                       save_path="/content/topik56_batch1000.jsonl",
                       max_attempts=20000,
                       checkpoint_every=100):
    """
    Sinh num_total mẫu, cân bằng theo (question_no in {5,6}) x (speaker in {'남자','여자'}).
    Checkpoint: ghi ra file JSONL mỗi khi đạt checkpoint_every mẫu mới (100 mặc định).
    """
    # quota 4 ô
    slots = [(5,"남자"), (5,"여자"), (6,"남자"), (6,"여자")]
    base = num_total // 4
    leftovers = num_total - base * 4
    target = {s: base for s in slots}
    for i in range(leftovers):
        target[slots[i % 4]] += 1

    got = Counter()
    rows = []            # giữ tất cả mẫu (để thống kê cuối)
    buffer = []          # chỉ các dòng mới kể từ checkpoint gần nhất

    # Xóa file cũ (nếu có) để tránh lẫn dữ liệu lần trước
    try:
        if os.path.exists(save_path):
            os.remove(save_path)
    except Exception:
        pass

    def _flush_buffer():
        """Append buffer ra file một cách an toàn rồi xóa buffer."""
        nonlocal buffer
        if not buffer:
            return
        # ghi ra file tạm trước rồi rename để tránh file dở dòng (atomic trên cùng filesystem)
        tmp_path = save_path + ".part"
        with open(tmp_path, "a", encoding="utf-8") as f:
            for line in buffer:
                f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        # gộp .part vào file chính theo kiểu append-atomic
        with open(tmp_path, "r", encoding="utf-8") as fin, open(save_path, "a", encoding="utf-8") as fout:
            for line in fin:
                fout.write(line)
            fout.flush()
            os.fsync(f.fileno())
            os.fsync(fout.fileno())
        # dọn file .part và làm rỗng buffer
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        buffer = []

    topic_cycle = itertools.cycle(topics)
    attempts = 0

    while sum(got.values()) < num_total and attempts < max_attempts:
        attempts += 1
        # ưu tiên ô còn thiếu nhiều nhất
        need_sorted = sorted(slots, key=lambda s: (target[s] - got[s]), reverse=True)
        qno, want_spk = next((q, sp) for (q, sp) in need_sorted if got[(q, sp)] < target[(q, sp)])

        topic = next(topic_cycle)
        body, _ = generate_one(topic=topic, qno=qno)
        ok, obj = validate_json(body)
        if not ok:
            continue

        spk = obj["dialogue"][0].get("speaker")
        if spk != want_spk:
            continue

        fixed = normalize_item(obj)
        rows.append(fixed)
        buffer.append(json.dumps(fixed, ensure_ascii=False))
        got[(qno, spk)] += 1

        total_now = sum(got.values())
        if total_now % 100 == 0:
            print(f"[PROG] {total_now}/{num_total} "
                  f"(5남:{got[(5,'남자')]}/{target[(5,'남자')]}, "
                  f"5여:{got[(5,'여자')]}/{target[(5,'여자')]}, "
                  f"6남:{got[(6,'남자')]}/{target[(6,'남자')]}, "
                  f"6여:{got[(6,'여자')]}/{target[(6,'여자')]})")

        # Checkpoint: ghi ra file mỗi khi đạt ngưỡng
        if len(buffer) >= checkpoint_every:
            _flush_buffer()

    # Ghi nốt phần còn lại
    _flush_buffer()

    print(f"[SAVE] {len(rows)} items (đã ghi dần) -> {save_path}")
    print("[TARGET]", target)
    print("[GOT]", dict(got))
    if len(rows) < num_total:
        print("[WARN] Not enough samples to meet quotas.",
              {k: target[k]-got[k] for k in target if got[k] < target[k]})

    # Thống kê nhanh
    dist_ans = Counter(r["answer"] for r in rows)
    dist_qno = Counter(r["question_no"] for r in rows)
    dist_spk = Counter(r["dialogue"][0]["speaker"] for r in rows)
    print("[DIST] answer:", dict(dist_ans))
    print("[DIST] qno:", dict(dist_qno))
    print("[DIST] speaker:", dict(dist_spk))
    return save_path

# ========= MAIN =========
if __name__ == "__main__":
    print("[INFO] Loading base + LoRA adapter from HF…")
    tok, model = load_model()

    print("[INFO] Generating one sample…")
    body, seed = generate_one(topic="shopping", qno=5)
    ok, obj_or_reason = validate_json(body)
    print("SEED:", seed)
    print("VALID:", ok)
    if ok:
        print(json.dumps(obj_or_reason, ensure_ascii=False, indent=2))
    else:
        print("REASON:", obj_or_reason)
        print("RAW OUT (truncated):", body[:1000])

  # ==== RUN: export 1000 balanced items to JSONL ====
    out_path = run_balanced_batch(
        num_total=1000,
        topics=("shopping","restaurant","transport","school","work","home","bank_post","social","travel_hotel","weather"),
        save_path="/content/topik56_batch1000.jsonl",   # Colab
        # save_path="/kaggle/working/topik56_batch1000.jsonl",  # Kaggle
        max_attempts=30000,
        checkpoint_every=100,
    )

    # (Tuỳ chọn) tải xuống nếu đang chạy Colab
    try:
        from google.colab import files
        files.download(out_path)
    except Exception:
        pass
