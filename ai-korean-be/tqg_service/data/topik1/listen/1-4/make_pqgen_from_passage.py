#!/usr/bin/env python3
import json, io, argparse, os

def iter_norm(path):
    with io.open(path,"r",encoding="utf-8") as f:
        for line in f: 
            if line.strip(): yield json.loads(line)

def ex_from_norm(obj):
    p = obj["passage"]["text_ko"].strip()
    q = obj["question"]
    return {
      "view": "pq_generate_from_passage",
      "id": obj["id"],
      "controls": {
        "section": obj["section"], "level": obj["level"],
        "mode": obj["passage"]["mode"], "num_questions": 1,
        "only_one_correct": True
      },
      "input": {"passage_ko": p},
      "assistant_output": {
        "questions": [{
          "prompt_ko": q["prompt_ko"],
          "choices": [{"label": c["label"], "text_ko": c.get("text_ko","")} for c in q["choices"]],
          "answer": q["answer"]["label"],
          "rationale_ko": q.get("rationale_ko","")
        }]
      }
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with io.open(args.out,"w",encoding="utf-8") as g:
        for o in iter_norm(args.inp):
            g.write(json.dumps(ex_from_norm(o), ensure_ascii=False)+"\n")

if __name__ == "__main__":
    main()
