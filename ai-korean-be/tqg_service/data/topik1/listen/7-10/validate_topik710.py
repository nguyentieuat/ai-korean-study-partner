#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validate JSONL cho TOPIK 7–10: schema + độ hợp lệ cơ bản.
"""
import json, sys
from collections import Counter

REQUIRED_KEYS = {"type","section","level","title","question","explanation","score","options","answer","dialogue","question_no"}
ANS_KEYS = {"A","B","C","D"}
PLACE_POOL = {
    "식당","시장","마트","빵집","카페","도서관","서점","학교","교실","회사","사무실","우체국",
    "은행","병원","약국","백화점","옷 가게","신발 가게","문구점","편의점","가구점",
    "사진관","미용실","호텔","공항","역","정류장","공원","박물관","영화관","미술관","수영장","운동장","가게"
}

def validate_file(path: str):
    ok = bad = 0
    ans_c = Counter(); qno_c = Counter(); spk_c = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[PARSE-ERR] line {ln}: {e}"); bad += 1; continue
            missing = REQUIRED_KEYS - obj.keys()
            if missing:
                print(f"[SCHEMA-ERR] line {ln}: missing {missing}"); bad += 1; continue
            opt = obj["options"]
            if set(opt.keys()) != ANS_KEYS:
                print(f"[OPT-ERR] line {ln}: options keys {set(opt.keys())}"); bad += 1; continue
            ans = obj["answer"]
            if ans not in ANS_KEYS:
                print(f"[ANS-ERR] line {ln}: answer {ans}"); bad += 1; continue
            dlg = obj["dialogue"]
            if not isinstance(dlg, list) or len(dlg) < 2:
                print(f"[DLG-ERR] line {ln}: dialogue must be >= 2 lines"); bad += 1; continue
            # speakers + text
            for turn in dlg:
                sp = turn.get("speaker","")
                if sp not in ("남자","여자"):
                    print(f"[SPK-ERR] line {ln}: speaker={sp}"); bad += 1; break
                if not str(turn.get("text","")).strip():
                    print(f"[TXT-ERR] line {ln}: empty text"); bad += 1; break
            else:
                qno = obj["question_no"]
                if qno not in (7,8,9,10):
                    print(f"[QNO-ERR] line {ln}: question_no={qno}"); bad += 1; continue
                # optional: check answer is plausible place
                if opt[ans] not in PLACE_POOL:
                    print(f"[PLACE-WARN] line {ln}: '{opt[ans]}' not in PLACE_POOL (skip fail).")
                ok += 1
                ans_c[ans] += 1; qno_c[qno] += 1; spk_c[dlg[0]["speaker"]] += 1
    print(f"[SUMMARY] OK={ok}, BAD={bad}")
    if ok > 0:
        print("[DISTRIBUTION] answer:", dict(ans_c))
        print("[DISTRIBUTION] question_no:", dict(qno_c))
        print("[DISTRIBUTION] speaker(first turn):", dict(spk_c))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: validate_topik710.py <jsonl_file>")
        sys.exit(1)
    validate_file(sys.argv[1])
