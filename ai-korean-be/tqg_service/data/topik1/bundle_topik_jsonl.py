
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, argparse
from pathlib import Path
from typing import List, Dict, Any

def read_jsonl(p: Path) -> List[Dict[str, Any]]:
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                out.append(json.loads(s))
            except Exception:
                # skip bad lines silently (assume normalized already)
                pass
    return out

def write_jsonl(p: Path, rows: List[Dict[str, Any]]):
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")

def bundle_every4(rows: List[Dict[str, Any]], prefix: str = "bundle") -> Dict[str, Any]:
    bundled = []
    report = {
        "total": len(rows),
        "bundles": 0,
        "full_bundles": 0,
        "partial_bundles": 0,
        "pattern_ok": 0,
        "pattern_bad": 0,
        "notes": []
    }
    i = 0
    bid = 0
    while i < len(rows):
        bid += 1
        group = rows[i:i+4]
        gid = f"{prefix}_{bid:05d}"
        for j, r in enumerate(group, start=1):
            r2 = dict(r)
            r2["bundle_id"] = gid
            r2["question_no"] = j
            bundled.append(r2)
        # validate score pattern if full
        if len(group) == 4:
            report["full_bundles"] += 1
            scores = [int(g.get("score", 0)) for g in group]
            c4 = sum(1 for s in scores if s == 4)
            c3 = sum(1 for s in scores if s == 3)
            if c4 == 2 and c3 == 2:
                report["pattern_ok"] += 1
            else:
                report["pattern_bad"] += 1
                report["notes"].append({"bundle_id": gid, "scores": scores})
        else:
            report["partial_bundles"] += 1
        report["bundles"] += 1
        i += 4
    return {"rows": bundled, "report": report}

def main():
    ap = argparse.ArgumentParser(description="Assign bundle_id & question_no (1..4) in file order for TOPIK I [1~4] style")
    ap.add_argument("--inp", required=True, help="Input JSONL (normalized)")
    ap.add_argument("--out", required=True, help="Output JSONL with bundle_id & question_no")
    ap.add_argument("--report", required=False, help="Write bundle report JSON")
    ap.add_argument("--prefix", default="bundle", help="Bundle id prefix (default: bundle)")
    args = ap.parse_args()

    inp = Path(args.inp)
    outp = Path(args.out)
    rows = read_jsonl(inp)
    res = bundle_every4(rows, prefix=args.prefix)
    write_jsonl(outp, res["rows"])
    if args.report:
        Path(args.report).write_text(json.dumps(res["report"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res["report"], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
