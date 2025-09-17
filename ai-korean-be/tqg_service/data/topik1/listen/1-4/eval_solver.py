# eval_solver.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import json, io, re, torch

MODEL_DIR = r"ckpts\topik_mcq\best"

def infer_one(m, tok, ex):
    prompt = (f"{ex['instruction_ko']}\n\n지문:\n{ex['context_ko']}\n\n보기:\n"
              + "\n".join(ex["choices"]) + "\n\n정답은 A/B/C/D 중 하나의 글자만 쓰세요.\n@@ANSWER@@ ")
    inpt = tok(prompt, return_tensors="pt").to(m.device)
    out  = m.generate(**inpt, max_new_tokens=1, do_sample=False)
    ans  = tok.decode(out[0][inpt['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    mch  = re.match(r"^[ABCD]", ans)
    return mch.group(0) if mch else "A"

m = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None, device_map="auto")
t = AutoTokenizer.from_pretrained(MODEL_DIR)

gold, pred = [], []
for line in io.open(r"data\topik1\1-4\splits\test.jsonl","r",encoding="utf-8"):
    ex = json.loads(line)
    pred.append(infer_one(m,t,ex)); gold.append(ex["label"])
acc = sum(p==g for p,g in zip(pred,gold))/len(gold)
print("TEST acc:", round(acc*100,2), "%")
