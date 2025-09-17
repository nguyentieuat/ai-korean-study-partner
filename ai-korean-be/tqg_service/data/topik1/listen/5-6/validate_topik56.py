#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validate JSONL for TOPIK 5–6 schema & distributions.
"""
import json, sys
from collections import Counter

REQUIRED_KEYS = {"type","section","level","title","question","explanation","score","options","answer","dialogue","question_no"}
ANS_KEYS = {"A","B","C","D"}

def validate_file(path: str):
    ok = 0
    bad = 0
    ans_counter = Counter()
    qno_counter = Counter()
    speaker_counter = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[PARSE-ERR] line {ln}: {e}")
                bad += 1
                continue
            missing = REQUIRED_KEYS - obj.keys()
            if missing:
                print(f"[SCHEMA-ERR] line {ln}: missing {missing}")
                bad += 1
                continue
            opt = obj["options"]
            if set(opt.keys()) != ANS_KEYS:
                print(f"[OPT-ERR] line {ln}: options keys {set(opt.keys())}")
                bad += 1
                continue
            ans = obj["answer"]
            if ans not in ANS_KEYS:
                print(f"[ANS-ERR] line {ln}: answer {ans}")
                bad += 1
                continue
            dlg = obj["dialogue"]
            if not isinstance(dlg, list) or len(dlg) != 1:
                print(f"[DLG-ERR] line {ln}: dialogue must be 1 line")
                bad += 1
                continue
            spk = dlg[0].get("speaker","")
            if spk not in ("남자","여자"):
                print(f"[SPK-ERR] line {ln}: speaker={spk}")
                bad += 1
                continue
            qno = obj["question_no"]
            if qno not in (5,6):
                print(f"[QNO-ERR] line {ln}: question_no={qno}")
                bad += 1
                continue
            ok += 1
            ans_counter[ans] += 1
            qno_counter[qno] += 1
            speaker_counter[spk] += 1

    print(f"[SUMMARY] OK={ok}, BAD={bad}")
    if ok > 0:
        print("[DISTRIBUTION] answer:", dict(ans_counter))
        print("[DISTRIBUTION] question_no:", dict(qno_counter))
        print("[DISTRIBUTION] speaker:", dict(speaker_counter))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: validate_topik56.py <jsonl_file>")
        sys.exit(1)
    validate_file(sys.argv[1])
