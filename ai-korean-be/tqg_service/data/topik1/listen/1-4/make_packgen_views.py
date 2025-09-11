#!/usr/bin/env python3
import json, io, argparse, re, os, random

def keyphrases_from_passage(text, k=3):
    # Trích từ đơn giản: lấy từ Hàn ≥2 kí tự, bỏ hư từ hay gặp
    stop = {"그리고","하지만","그래서","그러나","또한","이것","저것","오늘","내일"}
    toks = re.findall(r"[가-힣]{2,}", text)
    freq = {}
    for t in toks:
        if t in stop: continue
        freq[t] = freq.get(t,0)+1
    keys = sorted(freq, key=freq.get, reverse=True)[:k]
    return keys

def iter_norm(path):
    with io.open(path,"r",encoding="utf-8") as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def ex_from_norm(obj):
    p = obj["passage"]["text_ko"].strip()
    q = obj["question"]
    seeds = keyphrases_from_passage(p, k=3)
    return {
      "view": "pq_generate_full",
      "id": obj["id"],
      "controls": {
        "section": obj["section"], "level": obj["level"],
        "mode": obj["passage"]["mode"], "num_questions": 1,
        "length_passage_chars": [ max(60, min(220, len(p)-20)), min(280, len(p)+20) ],
        "only_one_correct": True
      },
      "seed": {
        "topic": "auto",
        "key_phrases_ko": seeds,
        "structure": ["시작/인사","핵심정보","마무리"] if obj["passage"]["mode"]!="dialogue"
                        else ["A발화","B응답"]
      },
      "assistant_output": {
        "passage_ko": p,
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
