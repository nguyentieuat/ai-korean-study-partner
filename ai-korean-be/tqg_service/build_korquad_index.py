#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust KorQuAD/kor QA corpus indexer (FAISS) — v3
- Hỗ trợ KorQuAD v1/v2 (SQuAD-like), JSON array, JSONL, và schema tổng quát có 'context'/'text'.
- Tuỳ chọn làm sạch HTML và loại bỏ các đoạn giống bảng HTML (table-like).
- Cho phép giới hạn số file, tăng batch size để tăng tốc.
- In log theo file khi bật --debug.

Outputs:
  outdir/faiss.index, outdir/texts.jsonl, outdir/meta.json
"""

import json, argparse, glob, os, re, sys
from html import unescape
from pathlib import Path
from typing import List, Dict, Any, Iterable, Set

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import torch


# ---------- Cleaning / Heuristics ----------

TAG_RE = re.compile(r"<[^>]+>")
TABLE_RE = re.compile(r"<\s*(table|tr|td|th|tbody|thead|tfoot)\b", re.I)

def looks_like_html_table(s: str) -> bool:
    """Heuristic: đoạn có thẻ bảng HTML."""
    if not isinstance(s, str):
        return False
    if TABLE_RE.search(s):
        return True
    # thêm chút heuristic phụ cho bảng kiểu wiki đã bỏ tag nhưng còn cột dạng ' | '
    if s.count("|") >= 5 and "\n" not in s:
        return True
    return False

def clean_text(s: str, strip_html: bool) -> str:
    """Unescape HTML entities, (tuỳ chọn) strip thẻ HTML, normalize khoảng trắng."""
    s = unescape(s or "")
    if strip_html:
        s = TAG_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------- Readers ----------

def _walk_contexts(obj: Any, min_chars_raw: int, seen: Set[str]) -> Iterable[str]:
    """Đệ quy tìm context trong dict/list theo schema phổ biến."""
    if isinstance(obj, dict):
        # KorQuAD/SQuAD style
        if "paragraphs" in obj and isinstance(obj["paragraphs"], list):
            for p in obj["paragraphs"]:
                ctx = p.get("context")
                if isinstance(ctx, str):
                    ctx = ctx.strip()
                    if len(ctx) >= min_chars_raw and ctx not in seen:
                        seen.add(ctx); yield ctx
        # Generic direct fields
        if "context" in obj and isinstance(obj["context"], str):
            c = obj["context"].strip()
            if len(c) >= min_chars_raw and c not in seen:
                seen.add(c); yield c
        if "text" in obj and isinstance(obj["text"], str):
            t = obj["text"].strip()
            if len(t) >= min_chars_raw and t not in seen:
                seen.add(t); yield t

        # Recurse
        for v in obj.values():
            yield from _walk_contexts(v, min_chars_raw, seen)

    elif isinstance(obj, list):
        for it in obj:
            yield from _walk_contexts(it, min_chars_raw, seen)


def extract_contexts_from_json_obj(obj: Any, min_chars_raw: int) -> List[str]:
    """Ưu tiên đi theo obj['data'] nếu có; fallback walk toàn bộ."""
    seen: Set[str] = set()
    out: List[str] = []
    if isinstance(obj, dict) and "data" in obj:
        for doc in obj.get("data", []):
            for c in _walk_contexts(doc, min_chars_raw, seen):
                out.append(c)
        if out:
            return out
    for c in _walk_contexts(obj, min_chars_raw, seen):
        out.append(c)
    return out


def read_any_json(fp: Path) -> List[Any]:
    """Đọc .json (object) hoặc .jsonl (mỗi dòng 1 object)."""
    if fp.suffix.lower() == ".jsonl":
        rows = []
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s: continue
                try:
                    rows.append(json.loads(s))
                except Exception:
                    pass
        return rows
    else:
        with fp.open("r", encoding="utf-8") as f:
            return [json.load(f)]


# ---------- Chunking & Embedding ----------

def chunk_text(s: str, max_chars: int, overlap: int) -> List[str]:
    out = []
    i, L = 0, len(s)
    while i < L:
        j = min(L, i + max_chars)
        out.append(s[i:j])
        if j == L: break
        i = max(0, j - overlap)
    return out


def make_embeddings(texts: List[str], model_name: str, batch_size: int) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    batch = [("passage: " + t) for t in texts]
    embs = model.encode(
        batch,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    return np.asarray(embs, dtype="float32")


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", nargs="+", required=True, help="JSON/JSONL file(s), supports glob")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--model", default="intfloat/multilingual-e5-base", help="SentenceTransformer model")
    ap.add_argument("--chunk-chars", type=int, default=250)
    ap.add_argument("--overlap", type=int, default=60)
    ap.add_argument("--min-context-chars", type=int, default=30, help="lọc sau khi sạch HTML (nếu bật strip)")
    ap.add_argument("--strip-html", action="store_true", help="gỡ thẻ HTML trước khi chunk")
    ap.add_argument("--skip-table", action="store_true", help="bỏ các context có bảng HTML (table-like)")
    ap.add_argument("--batch-size", type=int, default=64, help="batch size cho model.encode")
    ap.add_argument("--max-files", type=int, default=0, help="giới hạn số file đầu vào (0 = không giới hạn)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    # Gom danh sách file
    files: List[str] = []
    for pat in args.inp:
        files.extend(glob.glob(pat))
    files.sort()
    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        print("No input files matched for --inp.", file=sys.stderr); sys.exit(2)

    if args.debug:
        print(f"[DEBUG] matched {len(files)} files")
        for fp in files[:10]:
            print("  -", fp)
        print(f"[DEBUG] strip_html={args.strip_html}, skip_table={args.skip_table}, batch_size={args.batch_size}")

    texts: List[str] = []
    metas: List[Dict[str, Any]] = []
    total_ctx_raw = 0
    total_ctx_kept = 0

    for fpath in files:
        fp = Path(fpath)
        per_file_raw = 0
        per_file_kept = 0

        try:
            objs = read_any_json(fp)
        except Exception as e:
            if args.debug: print(f"[WARN] failed to read {fp}: {e}")
            continue

        for obj in objs:
            raw_ctxs = extract_contexts_from_json_obj(obj, min_chars_raw=1)  # lấy tất cả trước
            per_file_raw += len(raw_ctxs)

            for ctx in raw_ctxs:
                # Bỏ qua nếu yêu cầu skip bảng HTML
                if args.skip_table and looks_like_html_table(ctx):
                    continue

                # Làm sạch (unescape + strip HTML nếu bật)
                clean = clean_text(ctx, strip_html=args.strip_html)

                # Lọc theo độ dài sau khi làm sạch
                if len(clean) < args.min_context_chars:
                    continue

                # Chunking
                for ch in chunk_text(clean, args.chunk_chars, args.overlap):
                    metas.append({"id": len(texts), "text": ch, "source": fp.name})
                    texts.append(ch)
                    per_file_kept += 1

        total_ctx_raw += per_file_raw
        total_ctx_kept += per_file_kept
        if args.debug:
            print(f"[DEBUG] {fp.name}: raw_ctx={per_file_raw}, kept_chunks={per_file_kept}")

    if args.debug:
        print(f"[DEBUG] total raw contexts = {total_ctx_raw}")
        print(f"[DEBUG] total kept chunks = {len(texts)}")

    if not texts:
        print("No contexts extracted.", file=sys.stderr); sys.exit(3)

    # Embedding + FAISS index (cosine via IP on L2-normalized)
    X = make_embeddings(texts, args.model, args.batch_size)
    dim = int(X.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(X)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(outdir / "faiss.index"))
    with (outdir / "texts.jsonl").open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    with (outdir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump({
            "model_name": args.model,
            "dimension": dim,
            "count": int(X.shape[0]),
            "strip_html": bool(args.strip_html),
            "skip_table": bool(args.skip_table),
            "chunk_chars": int(args.chunk_chars),
            "overlap": int(args.overlap),
            "min_context_chars": int(args.min_context_chars),
            "batch_size": int(args.batch_size)
        }, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote index to {outdir} (chunks={len(texts)})")


if __name__ == "__main__":
    main()
