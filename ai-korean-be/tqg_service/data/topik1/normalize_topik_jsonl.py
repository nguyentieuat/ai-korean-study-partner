
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, re, sys, argparse, hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

P_SPACE = re.compile(r"\s+")
P_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,\.?!:;])")
P_SPACE_AFTER_PUNCT = re.compile(r"([,?!:;])\s*")
P_MULTI_DOTS = re.compile(r"\.{2,}")
P_TRAIL_SPACE = re.compile(r"\s+$")
P_LEAD_SPACE = re.compile(r"^\s+")

def norm_text(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = s.replace("\u00A0", " ")  # non-breaking space -> space
    s = P_LEAD_SPACE.sub("", s)
    s = P_TRAIL_SPACE.sub("", s)
    s = P_SPACE.sub(" ", s)
    s = P_SPACE_BEFORE_PUNCT.sub(r"\1", s)      # remove spaces before punctuation
    # Keep a single space after ,?!:; but not for final period ending Korean sentences necessarily
    s = P_SPACE_AFTER_PUNCT.sub(r"\1 ", s)
    s = P_MULTI_DOTS.sub(".", s)
    s = s.strip()
    return s

def norm_speaker(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = s.strip()
    if s.endswith(":"):
        s = s[:-1].strip()
    return s

def norm_dialogue(dlg: Any) -> List[Dict[str,str]]:
    out = []
    if isinstance(dlg, list):
        for t in dlg:
            spk = norm_speaker(t.get("speaker",""))
            txt = norm_text(t.get("text","") or "")
            if txt:
                out.append({"speaker": spk, "text": txt})
    return out

def ensure_score_int(v: Any, default=4) -> int:
    if isinstance(v, int):
        return v
    if isinstance(v, str) and v.strip().isdigit():
        return int(v.strip())
    return default

def options_ok(opts: Any) -> bool:
    if not isinstance(opts, dict):
        return False
    keys = set(opts.keys())
    needed = {"A","B","C","D"}
    if not needed.issubset(keys):
        return False
    for k in needed:
        if not isinstance(opts[k], str) or not opts[k].strip():
            return False
    return True

def answer_ok(ans: Any) -> bool:
    return isinstance(ans, str) and ans in {"A","B","C","D"}

def canonical_key(obj: Dict[str,Any]) -> str:
    # Build a stable key for dedup: (dialogue_text concatenated) + sorted options + answer
    dlg = obj.get("dialogue") or []
    dlg_text = " ".join([f'{t.get("speaker","")}:{t.get("text","")}' for t in dlg])
    opts = obj.get("options") or {}
    # Normalize each option text
    A = norm_text(opts.get("A","") or "")
    B = norm_text(opts.get("B","") or "")
    C = norm_text(opts.get("C","") or "")
    D = norm_text(opts.get("D","") or "")
    parts = [dlg_text, A, B, C, D, str(obj.get("answer"))]
    key = "||".join(parts)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()

def normalize_record(obj: Dict[str,Any]) -> Tuple[Optional[Dict[str,Any]], Optional[str]]:
    # Make a shallow copy
    rec = dict(obj)

    # Required fields minimal check
    if "options" not in rec or "answer" not in rec or "dialogue" not in rec:
        return None, "missing_required_fields"

    # Normalize dialogue
    rec["dialogue"] = norm_dialogue(rec.get("dialogue", []))
    if not rec["dialogue"]:
        return None, "empty_dialogue"

    # Normalize options
    if not isinstance(rec["options"], dict):
        return None, "options_not_dict"
    for k in ["A","B","C","D"]:
        v = rec["options"].get(k, "")
        rec["options"][k] = norm_text(v or "")
    if not options_ok(rec["options"]):
        return None, "options_invalid"

    # Normalize answer
    ans = rec.get("answer")
    if isinstance(ans, str):
        ans = ans.strip()
    rec["answer"] = ans
    if not answer_ok(rec["answer"]):
        return None, "answer_invalid"

    # Normalize score
    rec["score"] = ensure_score_int(rec.get("score", 4))

    # Normalize text fields
    for k in ["type","section","level","title","question","explanation","bundle_id","audio_url","topic","grammar_point"]:
        if k in rec and isinstance(rec[k], str):
            rec[k] = norm_text(rec[k])

    # Normalize question_no if any
    if "question_no" in rec:
        try:
            rec["question_no"] = int(rec["question_no"])
        except Exception:
            rec["question_no"] = None

    # Generate deterministic id
    rec["id"] = canonical_key(rec)

    return rec, None

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                yield i, json.loads(s)
            except Exception as e:
                yield i, {"__parse_error__": str(e), "__raw__": s}

def main():
    ap = argparse.ArgumentParser(description="Normalize & QC TOPIK MCQ JSONL (1 question = 1 JSON)")
    ap.add_argument("--inp", required=True, help="Input JSONL")
    ap.add_argument("--out", required=True, help="Output normalized JSONL")
    ap.add_argument("--report", required=False, help="QC report path (JSON)")
    args = ap.parse_args()

    inp = Path(args.inp)
    outp = Path(args.out)
    reportp = Path(args.report) if args.report else None

    seen_ids = set()
    n_in = n_ok = n_dup = n_bad = 0
    bad_reasons = {}

    with outp.open("w", encoding="utf-8") as fout:
        for ln, obj in iter_jsonl(inp):
            n_in += 1
            if "__parse_error__" in obj:
                n_bad += 1
                bad_reasons.setdefault("json_parse_error", 0)
                bad_reasons["json_parse_error"] += 1
                continue

            rec, err = normalize_record(obj)
            if err is not None:
                n_bad += 1
                bad_reasons.setdefault(err, 0)
                bad_reasons[err] += 1
                continue

            # Dedup
            cid = rec["id"]
            if cid in seen_ids:
                n_dup += 1
                continue
            seen_ids.add(cid)

            fout.write(json.dumps(rec, ensure_ascii=False)+"\n")
            n_ok += 1

    # Report
    rep = {
        "input": str(inp),
        "output": str(outp),
        "total_in": n_in,
        "total_ok": n_ok,
        "total_dup_skipped": n_dup,
        "total_bad": n_bad,
        "bad_reasons": bad_reasons
    }
    if reportp:
        with reportp.open("w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)

    print(json.dumps(rep, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
