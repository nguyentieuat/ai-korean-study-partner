#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, glob, io, json, os, re, csv, random
from typing import Dict, List, Tuple, Any

LETTER2IDX = {"A": 0, "B": 1, "C": 2, "D": 3}
IDX2LETTER = {v: k for k, v in LETTER2IDX.items()}

# -------------------- Light Korean linter rules --------------------

REPLACEMENTS = [
    # particle / spacing
    (r"동생와", "동생과"),
    (r"\b비\s*와요\b", "비가 와요"),
    (r"\b비가\s*와요\b", "비가 와요"),  # normalize variants
    # obvious semantic slip
    (r"어디에서\s*구두를\s*마셨어요\??", "어디에서 커피를 마셨어요?"),
]

QUESTION_CLUES = ("?", "몇", "어디", "누구", "왜", "언제", "어떻게", "무엇", "뭐", "어때요", "예요?", "이에요?")

def normalize_whitespace(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # keep newlines but collapse inner spaces
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" ?\n ?", "\n", s).strip()
    return s

def apply_fixes(s: str) -> str:
    if not isinstance(s, str) or not s:
        return s
    s = normalize_whitespace(s)
    for pat, repl in REPLACEMENTS:
        s = re.sub(pat, repl, s)
    # add final punctuation if likely statement (not a question)
    if s and not s.endswith((".", "!", "?", "…")):
        if not any(clue in s for clue in QUESTION_CLUES):
            s += "."
    return s

def join_dialogue(dialogue: List[Dict[str, str]]) -> Tuple[str, str]:
    if not dialogue:
        return "", "utterance"
    lines = []
    for turn in dialogue:
        spk = (turn.get("speaker") or "").strip()
        txt = (turn.get("text") or turn.get("text_ko") or "").strip()
        if not txt:
            continue
        line = f"{spk}: {txt}" if spk else txt
        lines.append(line)
    mode = "utterance" if len(lines) <= 1 else "dialogue"
    text = "\n".join(lines)
    return apply_fixes(text), mode

def map_section(sec: str) -> str:
    sec = (sec or "").strip()
    if sec in ("Nghe", "Listening"):
        return "Listening"
    if sec in ("Doc", "Đọc", "Reading"):
        return "Reading"
    return sec or "Listening"

def coerce_answer_label(ans: Any) -> str:
    """Accept 'A'/'a'/'  b  ' or 1..4."""
    if ans is None:
        return ""
    if isinstance(ans, (int, float)):
        i = int(ans) - 1
        return IDX2LETTER.get(i, "")
    s = str(ans).strip().upper()
    if s in LETTER2IDX:
        return s
    if s.isdigit():
        i = int(s) - 1
        return IDX2LETTER.get(i, "")
    return ""

def heuristic_rationale(passage: str, prompt: str, choices: List[Dict[str, str]], label: str, fallback: str) -> str:
    if fallback:
        return apply_fixes(fallback)
    p = (passage or "") + " " + (prompt or "")
    txt = ""
    for ch in choices:
        if ch.get("label") == label:
            txt = ch.get("text_ko") or ""
            break
    # simple pattern hints
    rules = [
        (r"몇\s*개", "수량 질문에 대한 수량 대답."),
        (r"몇\s*시|시간", "시간 질문에 대한 시간 대답."),
        (r"어디(에서|에)", "장소를 묻는 질문에 대한 장소 대답."),
        (r"누구(하고|와|과)|함께", "함께한 사람에 대한 대답."),
        (r"어때요|어떠", "상태/날씨 형용사로 대답."),
        (r"(예요\?|이에요\?)", "정체성/여부 질문에 대한 긍정/부정 대답."),
        (r"얼마|가격", "가격/비용에 대한 대답."),
        (r"왜", "이유를 묻는 질문에 대한 대답."),
        (r"어떻게", "방법을 묻는 질문에 대한 대답."),
        (r"무엇|뭐", "대상/사물을 묻는 질문에 대한 대답."),
    ]
    for pat, rat in rules:
        if re.search(pat, p):
            return rat
    if txt.endswith(("요.", "니다.", "예요.", "이에요.")):
        return "문맥에 가장 자연스러운 대답."
    return ""

# -------------------- Converters --------------------

def normalize_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    pack = raw.get("bundle_id") or raw.get("pack_id") or "TOPIK1_unknown_pack"
    qno = int(raw.get("question_no") or 0)
    norm_id = f"{pack}__{qno}"

    section = map_section(raw.get("section"))
    try:
        level = int(raw.get("level", 1))
    except Exception:
        level = 1

    passage_text, mode = join_dialogue(raw.get("dialogue") or [])
    prompt_ko = apply_fixes(raw.get("question") or raw.get("prompt") or "물음에 맞는 대답을 고르십시오.")

    # build choices from dict A..D
    opts = raw.get("options") or {}
    labels = ["A", "B", "C", "D"]
    choices = []
    for lab in labels:
        if lab not in opts:
            raise ValueError(f"missing option {lab}")
        choices.append({"label": lab, "text_ko": apply_fixes(str(opts[lab]))})

    ans_lab = coerce_answer_label(raw.get("answer"))
    if ans_lab not in LETTER2IDX:
        raise ValueError(f"invalid answer '{raw.get('answer')}'")
    ans_idx = LETTER2IDX[ans_lab]

    rationale = heuristic_rationale(
        passage_text, prompt_ko, choices, ans_lab,
        fallback=(raw.get("explanation") or "").strip()
    )

    out = {
        "version": "1.2",
        "id": norm_id,
        "pack_id": pack,
        "question_no": qno,
        "section": section,
        "level": level,
        "passage": {"mode": mode, "text_ko": passage_text},
        "question": {
            "type": "single_mcq",
            "prompt_ko": prompt_ko,
            "choices": choices,
            "answer": {"label": ans_lab, "index": ans_idx},
            "rationale_ko": rationale
        },
        "provenance": {
            "source": "topik_style_raw",
            "license": raw.get("license", "unknown"),
            "raw_id": raw.get("id"),
            "meta": {
                "title": raw.get("title"),
                "score": raw.get("score")
            }
        }
    }
    return out

def to_mcq_view(norm: Dict[str, Any], shuffle=False, seed=13) -> Dict[str, Any]:
    q = norm["question"]
    choices = q["choices"]
    lab = q["answer"]["label"]

    if shuffle:
        rng = random.Random(seed + hash(norm["id"]))
        perm = choices[:]
        rng.shuffle(perm)
        # remap label
        new_labels = ["A", "B", "C", "D"]
        # assign new labels in order
        remapped = []
        answer_label = None
        for i, ch in enumerate(perm):
            nl = new_labels[i]
            remapped.append(f"{nl}) {ch.get('text_ko') or ''}")
            if ch["label"] == lab:
                answer_label = nl
        lab = answer_label or lab
        choices_txt = remapped
    else:
        choices_txt = [f"{c['label']}) {c.get('text_ko') or ''}" for c in choices]

    return {
        "view": "mcq_answer",
        "id": norm["id"],
        "pack_id": norm["pack_id"],
        "section": norm["section"],
        "level": norm["level"],
        "instruction_ko": q["prompt_ko"] + " 정답은 A/B/C/D 중 하나만 출력하세요.",
        "context_ko": norm["passage"]["text_ko"],
        "choices": choices_txt,
        "label": lab,
        "rationale_ko": q.get("rationale_ko", "")
    }

# -------------------- IO --------------------

def read_jsonl(path: str):
    with io.open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise RuntimeError(f"{path}:{ln} JSON parse error: {e}")

def write_jsonl(path: str, items: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False))
            f.write("\n")

def write_report_csv(path: str, rows: List[Dict[str, Any]]):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with io.open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="Glob: data/topik1/1-4/dataset/topik_raw_*.jsonl")
    ap.add_argument("--out-normalized", required=True, help="Output source-of-truth v1.2 JSONL")
    ap.add_argument("--out-mcq", default="", help="(Optional) Output mcq_answer JSONL")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle choices when writing mcq view")
    ap.add_argument("--seed", type=int, default=13, help="Shuffle seed")
    ap.add_argument("--report-csv", default="", help="(Optional) path to data/topik1/1-4/dataset/reports/normalize_report.csv")
    ap.add_argument("--strict", action="store_true", help="Stop on first error")
    args = ap.parse_args()

    files = sorted(glob.glob(args.inp))
    if not files:
        raise SystemExit(f"No files matched: {args.inp}")

    normalized, mcq, report = [], [], []
    seen_ids = set()
    total, ok, skipped = 0, 0, 0

    for fp in files:
        for raw in read_jsonl(fp):
            total += 1
            raw_id = raw.get("id")
            try:
                norm = normalize_record(raw)
                if norm["id"] in seen_ids:
                    skipped += 1
                    report.append({"issue": "duplicate_norm_id", "norm_id": norm["id"], "raw_id": raw_id, "file": os.path.basename(fp)})
                    continue
                seen_ids.add(norm["id"])
                normalized.append(norm)
                ok += 1
                if args.out_mcq:
                    mcq.append(to_mcq_view(norm, shuffle=args.shuffle, seed=args.seed))
            except Exception as e:
                skipped += 1
                msg = str(e)
                report.append({"issue": "parse_fail", "raw_id": raw_id, "file": os.path.basename(fp), "error": msg})
                if args.strict:
                    raise

    # sort by pack & question_no to keep stable order
    normalized.sort(key=lambda x: (x["pack_id"], x.get("question_no", 0)))
    if args.out_mcq:
        mcq.sort(key=lambda x: (x["pack_id"], x["id"]))

    write_jsonl(args.out_normalized, normalized)
    if args.out_mcq:
        write_jsonl(args.out_mcq, mcq)
    if args.report_csv:
        write_report_csv(args.report_csv, report)

    print(f"Done. files={len(files)} total={total} ok={ok} skipped={skipped}")
    print(f"  normalized -> {args.out_normalized}")
    if args.out_mcq:
        print(f"  mcq_view   -> {args.out_mcq}")
    if args.report_csv:
        print(f"  report     -> {args.report_csv}  (rows={len(report)})")

if __name__ == "__main__":
    main()
