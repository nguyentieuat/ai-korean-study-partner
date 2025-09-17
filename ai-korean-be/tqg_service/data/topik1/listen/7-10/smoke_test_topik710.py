#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Smoke test for TOPIK I (7~10) generator checkpoints.
# - Loads LoRA/QLoRA adapter (PEFT) or merged model.
# - Generates N samples across 20 topics.
# - Extracts JSON, validates schema & basic semantics.
# - Writes outputs (JSONL) and a short report.

import os, sys, re, json, time, argparse, random
from typing import List, Dict, Any, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PLACE_POOL = [
    "식당","시장","마트","빵집","카페","도서관","서점","학교","교실","회사","사무실","우체국",
    "은행","병원","약국","백화점","옷 가게","신발 가게","문구점","편의점","가구점",
    "사진관","미용실","호텔","공항","역","정류장","공원","박물관","영화관","미술관","수영장","운동장","가게"
]

TOPICS = [
    "식당","시장/마트","빵집/카페","도서관","서점","학교/교실","회사/사무실","우체국","은행",
    "병원/약국","백화점/옷 가게/신발 가게","미용실","사진관","호텔","공항/역/정류장","공원",
    "박물관/영화관/미술관","운동장/수영장","가구점/문구점","편의점"
]

SYSTEM = "You are a Korean TOPIK question writer who returns only valid JSON without extra text."
INSTR = (
    "TOPIK I (초급) 7~10 유형의 ‘여기는 어디입니까?’ 문제를 '처음부터' 생성하세요. 출력은 JSON 한 개만.\n"
    "키/제약:\n"
    "- options: A,B,C,D (서로 달라야 함),\n"
    "- answer: A|B|C|D (options 키 중 하나),\n"
    "- dialogue: [ { speaker: 남자|여자, text: ... }, ... ] 2~6줄,\n"
    "- question_no: 7|8|9|10,\n"
    "- JSON 외의 텍스트 출력 금지.\n"
)
RESPONSE_TAG = "#### JSON_START ####"

def build_user_prompt(topic=None, qno=None):
    base = "TOPIK I [7~10] 유형으로 ‘여기는 어디입니까?’ 문제를 1개 생성하세요. JSON으로만 출력하세요."
    if topic:
        base += f" 주제(장소): {topic}."
    if qno in (7,8,9,10):
        base += f" question_no: {qno}."
    base += " 필드: type, section, level, title, question, explanation, score, options, answer, dialogue, question_no."
    base += " 정답은 A~D 중 하나입니다."
    return base

def to_chat(tokenizer, system, user):
    try:
        return tokenizer.apply_chat_template(
            [{"role":"system","content":system},{"role":"user","content":user}],
            tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback: simple format
        return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"

def try_load_model(ckpt: str, base_model: str = None):
    # Try to load as PEFT adapter first; if fails, try merged checkpoint.
    tok = AutoTokenizer.from_pretrained(ckpt, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    try:
        print("[INFO] Loading as PEFT adapter:", ckpt)
        mdl = AutoPeftModelForCausalLM.from_pretrained(ckpt, device_map="auto", torch_dtype="auto", trust_remote_code=True)
        return tok, mdl
    except Exception as e:
        print("[WARN] AutoPeft load failed:", e)
        if base_model is None:
            print("[INFO] Trying to load directly as merged/vanilla:", ckpt)
            mdl = AutoModelForCausalLM.from_pretrained(ckpt, device_map="auto", torch_dtype="auto", trust_remote_code=True)
            return tok, mdl
        else:
            print("[INFO] Falling back to base model + (assumed merged tokenizer):", base_model)
            tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
            if tok.pad_token is None: tok.pad_token = tok.eos_token
            mdl = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype="auto", trust_remote_code=True)
            # NOTE: If 'ckpt' is PEFT-only and base_model is provided, user should merge Adapters beforehand.
            return tok, mdl

def gen_one(tokenizer, model, topic=None, qno=None, temperature=0.8, max_new_tokens=400):
    user = build_user_prompt(topic=topic, qno=qno)
    prompt = to_chat(tokenizer, SYSTEM, user)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            repetition_penalty=1.05
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # Try to extract last JSON object
    l = text.rfind("{")
    r = text.rfind("}")
    if l != -1 and r != -1 and r > l:
        cand = text[l:r+1]
    else:
        cand = text
    return cand, text

def validate_js(obj: Dict[str, Any]) -> Tuple[bool, str]:
    REQ = {"type","section","level","title","question","explanation","score","options","answer","dialogue","question_no"}
    if not isinstance(obj, dict):
        return False, "not a dict"
    miss = REQ - set(obj.keys())
    if miss:
        return False, f"missing keys: {sorted(miss)}"
    if not isinstance(obj["options"], dict) or sorted(obj["options"].keys()) != ["A","B","C","D"]:
        return False, "options must have keys A,B,C,D"
    if obj["answer"] not in ("A","B","C","D"):
        return False, "invalid answer key"
    # distinct options
    optvals = list(obj["options"].values())
    if len(set(optvals)) != 4:
        return False, "duplicate option values"
    # dialogue
    dlg = obj["dialogue"]
    if not isinstance(dlg, list) or len(dlg) < 2:
        return False, "dialogue too short"
    for t in dlg:
        if t.get("speaker") not in ("남자","여자"):
            return False, f"bad speaker: {t.get('speaker')}"
        if not str(t.get("text","")).strip():
            return False, "empty dialogue text"
    # question_no
    if obj["question_no"] not in (7,8,9,10):
        return False, f"bad question_no: {obj['question_no']}"
    # score consistency: 10 -> 4, else -> 3  (soft fix if provided wrong)
    want = 4 if obj["question_no"] == 10 else 3
    if obj["score"] != want:
        obj["score"] = want  # repair softly
    # place plausibility
    if obj["options"][obj["answer"]] not in PLACE_POOL:
        return False, f"answer '{obj['options'][obj['answer']]}' not in PLACE_POOL"
    return True, "ok"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Checkpoint dir (PEFT adapter or merged).")
    ap.add_argument("--base_model", default=None, help="Base model for fallback (optional).")
    ap.add_argument("--num", type=int, default=40, help="Total generations (default: 40)")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--outfile", default="smoke_topik710_outputs.jsonl")
    ap.add_argument("--report", default="smoke_topik710_report.txt")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok, mdl = try_load_model(args.ckpt, base_model=args.base_model)
    mdl.eval()

    topics_cycle = (TOPICS * ((args.num + len(TOPICS) - 1)//len(TOPICS)))[:args.num]
    qnos_cycle = [7,8,9,10] * ((args.num + 3)//4)
    out_ok = []
    out_fail = []
    st = time.time()

    for i in range(args.num):
        topic = topics_cycle[i]
        qno = qnos_cycle[i]
        raw_json_str, full_text = gen_one(tok, mdl, topic=topic, qno=qno, temperature=args.temperature)
        try:
            js = json.loads(raw_json_str)
        except Exception as e:
            # try naive trim
            try:
                raw2 = raw_json_str[raw_json_str.find("{"): raw_json_str.rfind("}")+1]
                js = json.loads(raw2)
            except Exception as e2:
                out_fail.append({"i": i, "topic": topic, "qno": qno, "error": f"parse_error: {e2}", "raw": full_text[-800:]})
                continue
        ok, msg = validate_js(js)
        if ok:
            out_ok.append(js)
        else:
            out_fail.append({"i": i, "topic": topic, "qno": qno, "error": msg, "raw_json": js})

    dt = time.time()-st
    os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
    with open(args.outfile, "w", encoding="utf-8") as f:
        for r in out_ok:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(args.report, "w", encoding="utf-8") as f:
        f.write(f"# TOPIK 7-10 Smoke Test Report\n")
        f.write(f"- ckpt: {args.ckpt}\n")
        if args.base_model: f.write(f"- base_model: {args.base_model}\n")
        f.write(f"- num: {args.num}\n")
        f.write(f"- temperature: {args.temperature}\n")
        f.write(f"- time: {dt:.2f}s\n")
        f.write(f"- pass: {len(out_ok)} / {args.num}  (rate={len(out_ok)/max(1,args.num):.2%})\n")
        f.write(f"- fail: {len(out_fail)}\n")
        if out_fail:
            f.write("\n## Fail samples (up to 10):\n")
            for e in out_fail[:10]:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"[DONE] pass={len(out_ok)}/{args.num}  rate={len(out_ok)/max(1,args.num):.2%},  took {dt:.2f}s")
    print(f"[OUT] JSONL: {args.outfile}")
    print(f"[OUT] Report: {args.report}")

if __name__ == "__main__":
    main()
