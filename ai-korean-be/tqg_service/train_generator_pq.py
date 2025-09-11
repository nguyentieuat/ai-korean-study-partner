# train_generator_pq.py
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, PeftModel
import json, io, torch, os

BASE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUT_ADAPTER = r"ckpts\topik_gen\adapter"
OUT_MERGED  = r"ckpts\topik_gen\best"
TAG = "@@JSON@@"

def load_jsonl(p):
    with io.open(p,"r",encoding="utf-8") as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def prompt_from_passage(ex):
    ctrl = ex["controls"]
    inp  = ex["input"]["passage_ko"]
    out  = {
      "question_ko": ex["assistant_output"]["questions"][0]["prompt_ko"],
      "choices": ex["assistant_output"]["questions"][0]["choices"],
      "answer": ex["assistant_output"]["questions"][0]["answer"],
      "rationale_ko": ex["assistant_output"]["questions"][0].get("rationale_ko","")
    }
    instr = (f"<SECTION={ctrl['section']}> <LEVEL={ctrl['level']}> <MODE={ctrl['mode']}> "
             f"<NUMQ={ctrl['num_questions']}> <ONLYONE={int(ctrl['only_one_correct'])}>\n"
             "지문을 읽고 TOPIK 스타일의 4지선다 문항 1개를 JSON으로 생성하세요.\n"
             f"지문:\n{inp}\n\n{TAG}\n")
    return {"text": instr + json.dumps(out, ensure_ascii=False)}

def prompt_full(ex):
    ctrl = ex["controls"]
    out  = ex["assistant_output"]  # đã có passage_ko + questions
    instr = (f"<SECTION={ctrl['section']}> <LEVEL={ctrl['level']}> <MODE={ctrl['mode']}> "
             f"<NUMQ={ctrl['num_questions']}> <LEN={ctrl['length_passage_chars']}> "
             f"<ONLYONE={int(ctrl['only_one_correct'])}>\n"
             "TOPIK 스타일로 지문(또는 대화)과 4지선다 문항 1개를 JSON으로 생성하세요.\n"
             f"{TAG}\n")
    return {"text": instr + json.dumps(out, ensure_ascii=False)}

train = []
for ex in load_jsonl(r"data\topik1\1-4\dataset_pqgen_views\from_passage.jsonl"):
    train.append(prompt_from_passage(ex))
for ex in load_jsonl(r"data\topik1\1-4\dataset_packgen_views\full.jsonl"):
    train.append(prompt_full(ex))

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
collator = DataCollatorForCompletionOnlyLM(response_template=TAG, tokenizer=tok)

peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
args = TrainingArguments(
    output_dir=OUT_ADAPTER, per_device_train_batch_size=2, gradient_accumulation_steps=16,
    learning_rate=8e-5, num_train_epochs=2, warmup_ratio=0.05, logging_steps=50,
    evaluation_strategy="no", save_steps=1000, bf16=torch.cuda.is_available(), report_to="none"
)

model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None)
trainer = SFTTrainer(model=model, tokenizer=tok, args=args,
                     peft_config=peft_cfg, data_collator=collator,
                     train_dataset=train, dataset_text_field="text")
trainer.train()
trainer.model.save_pretrained(OUT_ADAPTER)
tok.save_pretrained(OUT_ADAPTER)

# MERGE
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None)
from peft import PeftModel
merged = PeftModel.from_pretrained(base, OUT_ADAPTER).merge_and_unload()
os.makedirs(OUT_MERGED, exist_ok=True)
merged.save_pretrained(OUT_MERGED)
tok.save_pretrained(OUT_MERGED)
print("Saved merged generator to", OUT_MERGED)
