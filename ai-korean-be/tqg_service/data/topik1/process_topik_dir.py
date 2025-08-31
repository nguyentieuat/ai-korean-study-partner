
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, subprocess, argparse, os
from pathlib import Path

def find_files(root: Path, pattern: str, recursive: bool):
    if recursive:
        yield from root.rglob(pattern)
    else:
        yield from root.glob(pattern)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr

def main():
    ap = argparse.ArgumentParser(description="Normalize + Bundle all TOPIK JSONL files (safe in-place)")
    ap.add_argument("--root", default=".", help="Root folder to scan (default: .)")
    ap.add_argument("--pattern", default="*.jsonl", help="Filename pattern (default: *.jsonl)")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirs")
    ap.add_argument("--normalize-script", default="data/topik1/normalize_topik_jsonl.py", help="Path to normalize_topik_jsonl.py")
    ap.add_argument("--bundle-script", default="data/topik1/bundle_topik_jsonl.py", help="Path to bundle_topik_jsonl.py")
    ap.add_argument("--prefix", default="TOPIK1", help="Bundle id prefix (default: TOPIK1)")
    ap.add_argument("--dry-run", action="store_true", help="List files without processing")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    norm_script = Path(args.normalize_script)
    bund_script = Path(args.bundle_script)

    if not norm_script.exists():
        print(f"[ERR] normalize script not found: {norm_script}", file=sys.stderr)
        sys.exit(2)
    if not bund_script.exists():
        print(f"[ERR] bundle script not found: {bund_script}", file=sys.stderr)
        sys.exit(2)

    files = sorted(find_files(root, args.pattern, args.recursive))
    if not files:
        print("[INFO] No files matched.")
        return

    print(f"[INFO] Found {len(files)} file(s).")
    processed, failed = 0, 0
    for f in files:
        # skip likely generated files (optional)
        if f.name.endswith(".tmp"):
            continue

        print(f"\n==> {f}")
        if args.dry_run:
            continue

        tmp = f.with_suffix(f.suffix + ".tmp")
        repdir = f.parent / "reports"
        ensure_dir(repdir)
        norm_rep = repdir / (f.stem + "_normalize.json")
        bund_rep = repdir / (f.stem + "_bundle.json")

        # 1) Normalize
        cmd1 = [sys.executable, str(norm_script), "--inp", str(f), "--out", str(tmp), "--report", str(norm_rep)]
        rc, out, err = run(cmd1)
        print(out.strip())
        if rc != 0:
            print(f"[ERR] normalize failed: {f}\n{err}", file=sys.stderr)
            failed += 1
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            continue
        # replace in-place
        os.replace(tmp, f)

        # 2) Bundle
        file_prefix = f"{args.prefix}_{f.stem}"
        cmd2 = [sys.executable, str(bund_script), "--inp", str(f), "--out", str(tmp), "--report", str(bund_rep), "--prefix", file_prefix]
        rc, out, err = run(cmd2)
        print(out.strip())
        if rc != 0:
            print(f"[ERR] bundle failed: {f}\n{err}", file=sys.stderr)
            failed += 1
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            continue
        os.replace(tmp, f)

        processed += 1

    print(f"\n[SUM] processed={processed}, failed={failed}, total={len(files)}")

if __name__ == "__main__":
    main()
