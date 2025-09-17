#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demo router: load base once, mount 2 adapters (1–4 and 5–6), choose by question_no
"""
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

INSTR = """### Instruction:
당신은 TOPIK I (초급) 5~6 유형 문제를 생성하는 모델입니다.
아래의 한 줄 대화를 보고, 이어지는 말을 고르는 JSON을 출력하세요.
출력은 반드시 JSON 한 개만 포함해야 합니다.
"""

PROMPT_TMPL = """<|begin_of_text|>
{instr}

### Input:
{dialogue_line}

### Response:
"""

def build_prompt(dialogue_line: str) -> str:
    return PROMPT_TMPL.format(instr=INSTR, dialogue_line=dialogue_line)

def load_model_and_adapters(base:str, adapter_14:str, adapter_56:str):
    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, device_map="auto")
    model = PeftModel.from_pretrained(model, adapter_14, adapter_name="topik14")
    model.load_adapter(adapter_56, adapter_name="topik56")
    return tok, model

def generate_item(tok, model, dialogue_line: str, task="topik56"):
    model.set_adapter(task)
    prompt = build_prompt(dialogue_line)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=220, do_sample=True, temperature=0.7, top_p=0.9)
    text = tok.decode(out[0], skip_special_tokens=True)
    if "### Response:" in text:
        text = text.split("### Response:")[-1].strip()
    return text

if __name__ == "__main__":
    BASE = os.environ.get("BASE_MODEL","meta-llama/Meta-Llama-3-8B-Instruct")
    ADP14 = os.environ.get("ADAPTER_14","nguyentieuat/topik-mcq-llama3-8b-lora")
    ADP56 = os.environ.get("ADAPTER_56","nguyentieuat/topik-56-llama3-8b-lora")
    tok, model = load_model_and_adapters(BASE, ADP14, ADP56)
    print(generate_item(tok, model, "여자: 우산을 잃어버렸어요.", task="topik56"))
