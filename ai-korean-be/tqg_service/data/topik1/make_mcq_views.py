
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, argparse, random
from typing import List, Dict, Any, Tuple
from pathlib import Path
from collections import defaultdict
import glob

def read_jsonl_many(paths: List[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    rows.append(json.loads(s))
                except Exception:
                    pass
    return rows

def group_by_bundle(rows: List[Dict[str,Any]]) -> Dict[str, List[Dict[str,Any]]]:
    buckets: Dict[str, List[Dict[str,Any]]] = defaultdict(list)
    for r in rows:
        bid = r.get("bundle_id") or "__no_bundle__"
        buckets[bid].append(r)
    # sort each group by question_no if present, else keep order
    for bid,g in buckets.items():
        if all("question_no" in x for x in g):
            g.sort(key=lambda x: x["question_no"])
    return buckets

def make_prompt(r: Dict[str,Any]) -> str:
    title = r.get("title") or "[1~4] 다음을 듣고 물음에 맞는 대답을 고르십시오."
    q = r.get("question") or "물음에 맞는 대답을 고르십시오."
    dlg = r.get("dialogue") or []
    if dlg:
        turn = dlg[0]
        s = f'{turn.get("speaker","")}: {turn.get("text","")}'
    else:
        s = ""
    opts = r.get("options", {})
    A,B,C,D = opts.get("A",""), opts.get("B",""), opts.get("C",""), opts.get("D","")
    prompt = (
        f"{title}\n"
        f"{q}\n"
        f"질문: {s}\n"
        f"선택지:\n"
        f"A. {A}\n"
        f"B. {B}\n"
        f"C. {C}\n"
        f"D. {D}\n"
        f"정답의 글자만 출력하세요."
    ).strip()
    return prompt

def to_classification(r: Dict[str,Any]) -> Dict[str,Any]:
    dlg = r.get("dialogue") or []
    if dlg:
        s = f'{dlg[0].get("speaker","")}: {dlg[0].get("text","")}'
    else:
        s = ""
    inp = f'{r.get("title","")} {r.get("question","")} 질문: {s}'.strip()
    opts = r.get("options", {})
    choices = [opts.get("A",""), opts.get("B",""), opts.get("C",""), opts.get("D","")]
    label = {"A":0,"B":1,"C":2,"D":3}[r["answer"]]
    meta = {
        "id": r.get("id"),
        "bundle_id": r.get("bundle_id"),
        "question_no": r.get("question_no"),
        "score": int(r.get("score",0))
    }
    return {"input": inp, "choices": choices, "label": label, "meta": meta}

def to_sft(r: Dict[str,Any]) -> Dict[str,Any]:
    return {"prompt": make_prompt(r), "response": r["answer"]}

def split_bundles(buckets: Dict[str, List[Dict[str,Any]]], val_ratio=0.1, test_ratio=0.1, seed=42, full_only=True):
    items = list(buckets.items())
    if full_only:
        items = [(bid,g) for bid,g in items if len(g)==4]
    rng = random.Random(seed)
    rng.shuffle(items)
    n = len(items)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test_items = items[:n_test]
    val_items = items[n_test:n_test+n_val]
    train_items = items[n_test+n_val:]

    def flatten(groups):
        out = []
        for _,g in groups:
            out.extend(g)
        return out

    return {
        "train": flatten(train_items),
        "val": flatten(val_items),
        "test": flatten(test_items),
        "counts": {"bundles_total": len(buckets), "bundles_used": len(items),
                   "train_bundles": len(train_items), "val_bundles": len(val_items), "test_bundles": len(test_items)}
    }

def write_jsonl(p: Path, rows: List[Dict[str,Any]]):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")

def main():
    ap = argparse.ArgumentParser(description="Build train/val/test splits (by bundle) and export classification & SFT JSONL views.")
    ap.add_argument("--inp", nargs="+", required=True, help="Input JSONL file(s) (normalized + bundled)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-partials", action="store_true", help="Include bundles with size != 4 (default: drop)")
    args = ap.parse_args()

    expanded = []
    for pat in args.inp:
        # hỗ trợ cả ** nếu bạn muốn dùng đệ quy
        expanded.extend(glob.glob(pat, recursive=True))
    paths = [Path(p) for p in expanded]
    rows = read_jsonl_many(paths)
    buckets = group_by_bundle(rows)

    res = split_bundles(buckets, val_ratio=args.val_ratio, test_ratio=args.test_ratio,
                        seed=args.seed, full_only=not args.include_partials)

    # Build views
    splits = {}
    for split_name, split_rows in [("train", res["train"]), ("val", res["val"]), ("test", res["test"])]:
        cls_rows = [to_classification(r) for r in split_rows]
        sft_rows = [to_sft(r) for r in split_rows]
        splits[split_name] = {"classification": cls_rows, "sft": sft_rows}

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for split_name, payload in splits.items():
        write_jsonl(outdir / f"{split_name}.classification.jsonl", payload["classification"])
        write_jsonl(outdir / f"{split_name}.sft.jsonl", payload["sft"])

    # summary
    stats = {
        "input_files": [str(p) for p in paths],
        "counts": res["counts"],
        "sizes": {k: {"samples": len(v["classification"]), "sft_samples": len(v["sft"])} for k,v in splits.items()}
    }
    (outdir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
