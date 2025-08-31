#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Augment SFT JSONL ({"prompt","response"}) bằng cách truy hồi top-K KorQuAD chunks
và chèn block '참고 지문' vào cuối prompt.

VÁ: tự khớp model embedding theo index
- Đọc meta.json trong --indexdir để lấy "model_name" và "dimension"
- Nếu --model khác meta["model_name"], sẽ cảnh báo và dùng model từ index
- Kiểm tra dimension của embedding query == index.d trước khi search

Debug options:
  --debug                 : In top-K (score, source, snippet) cho từng mẫu
  --debug-limit N         : Chỉ in debug cho N mẫu đầu (mặc định 10)
  --debug-snippet N       : Độ dài tối đa của snippet khi in debug
  --save-log path.tsv     : Ghi toàn bộ kết quả truy hồi ra TSV
"""

import json
import argparse
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


# ----------------- I/O helpers -----------------

def read_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield json.loads(s)

def get_texts_and_sources(indexdir: Path):
    texts, sources = [], []
    with (indexdir / "texts.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            texts.append(o["text"])
            sources.append(o.get("source", ""))
    return texts, sources

def load_index(indexdir: Path):
    index = faiss.read_index(str(indexdir / "faiss.index"))
    meta = json.loads((indexdir / "meta.json").read_text(encoding="utf-8"))
    texts, sources = get_texts_and_sources(indexdir)
    return index, meta, texts, sources


# ----------------- Embedding & search -----------------

def build_query_encoder(model_name: str):
    # SentenceTransformer sẽ tự chọn CPU/GPU theo torch.cuda.is_available()
    return SentenceTransformer(model_name)

def extract_question_from_prompt(prompt: str) -> str:
    """
    Cố gắng lấy phần sau '질문:' làm query. Nếu không có thì dùng toàn bộ prompt.
    """
    m = re.search(r"질문:\s*(.*)", prompt)
    if m:
        return m.group(1).strip()
    return prompt.strip()

def encode_queries(encoder, queries: List[str]) -> np.ndarray:
    # e5-family khuyến nghị prefix 'query: ' cho truy vấn
    batch = [("query: " + q) for q in queries]
    X = encoder.encode(batch, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(X, dtype="float32")

def search(index, Xq: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    # FAISS IP trên vector đã chuẩn hoá ~ cosine similarity
    scores, ids = index.search(Xq, topk)
    return scores, ids


# ----------------- Main -----------------

def main():
    ap = argparse.ArgumentParser(description="Augment SFT với top-K KorQuAD contexts (auto-match model by index)")
    ap.add_argument("--sft", required=True, help="Input SFT JSONL (prompt/response)")
    ap.add_argument("--indexdir", required=True, help="Thư mục chứa faiss.index, texts.jsonl, meta.json")
    ap.add_argument("--out", required=True, help="Output JSONL")
    ap.add_argument("--model", default=None, help="(Tùy chọn) Embedding model; sẽ bị override bởi meta.json nếu khác")
    ap.add_argument("--topk", type=int, default=3)
    # Debug options
    ap.add_argument("--debug", action="store_true", help="In top-K (score, source, snippet) cho từng mẫu")
    ap.add_argument("--debug-limit", type=int, default=10, help="Chỉ in debug cho N mẫu đầu")
    ap.add_argument("--debug-snippet", type=int, default=120, help="Số ký tự tối đa của snippet khi in debug")
    ap.add_argument("--save-log", type=str, default=None, help="Ghi toàn bộ kết quả truy hồi ra TSV")
    args = ap.parse_args()

    indexdir = Path(args.indexdir)
    index, meta, texts, sources = load_index(indexdir)

    # --- Auto-match model by index meta ---
    index_model = meta.get("model_name")
    index_dim = int(meta.get("dimension", index.d)) if isinstance(meta.get("dimension", None), (int, float)) else index.d

    chosen_model = index_model or args.model or "intfloat/multilingual-e5-base"
    if args.model and index_model and args.model != index_model:
        print(f"[WARN] --model ({args.model}) ≠ index model ({index_model}); dùng model theo index: {index_model}")
        chosen_model = index_model

    encoder = build_query_encoder(chosen_model)

    # safety check: dimension khớp index?
    probe = encoder.encode(["query: sanity"], show_progress_bar=False, normalize_embeddings=True)
    if probe.ndim != 2:
        raise RuntimeError(f"Probe embedding shape bất thường: {probe.shape}")
    if probe.shape[1] != index_dim:
        raise RuntimeError(
            f"Embedding dim {probe.shape[1]} ≠ index dim {index_dim}. "
            f"Hãy dùng model {index_model} hoặc rebuild index cho khớp."
        )

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    # Gom queries để encode 1 lượt
    prompts, examples = [], []
    for ex in read_jsonl(Path(args.sft)):
        q = extract_question_from_prompt(ex["prompt"])
        prompts.append(q)
        examples.append(ex)

    Xq = encode_queries(encoder, prompts)
    scores, ids = search(index, Xq, args.topk)

    # (tuỳ chọn) mở file log TSV
    log_fh = None
    if args.save_log:
        log_path = Path(args.save_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = log_path.open("w", encoding="utf-8")
        log_fh.write("row\tquery\trank\tscore\tsource\tsnippet\n")

    with outp.open("w", encoding="utf-8") as fout:
        for row_id, (ex, nn, scs) in enumerate(zip(examples, ids, scores)):
            # Build ctxs
            ctxs = [texts[sid] for sid in nn if sid >= 0]

            # Compose prompt
            ctx_block = ""
            if ctxs:
                ctx_block = "참고 지문:\n" + "\n".join([f"- {c}" for c in ctxs])

            new_prompt = ex["prompt"] + ("\n\n" + ctx_block if ctx_block else "")
            out = {"prompt": new_prompt, "response": ex["response"]}
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

            # Debug printing
            if args.debug and row_id < args.debug_limit:
                q_show = prompts[row_id]
                print(f"[DEBUG row={row_id}] query: {q_show}")
                for rank, (sid, sc) in enumerate(zip(nn, scs), start=1):
                    if sid < 0: continue
                    src = sources[sid] if sid < len(sources) else ""
                    snippet = texts[sid].replace("\n", " ")
                    if len(snippet) > args.debug_snippet:
                        snippet = snippet[:args.debug_snippet] + "…"
                    print(f"  {rank}. score={sc:.3f} source={src} | {snippet}")

            # Optional log
            if log_fh is not None:
                q_show = prompts[row_id].replace("\t", " ").replace("\n", " ")
                for rank, (sid, sc) in enumerate(zip(nn, scs), start=1):
                    if sid < 0: continue
                    src = sources[sid] if sid < len(sources) else ""
                    snippet = texts[sid].replace("\n", " ")
                    if len(snippet) > args.debug_snippet:
                        snippet = snippet[:args.debug_snippet] + "…"
                    log_fh.write(f"{row_id}\t{q_show}\t{rank}\t{sc:.6f}\t{src}\t{snippet}\n")

    if log_fh is not None:
        log_fh.close()

    print(f"[OK] wrote {outp}\n[INFO] encoder_model={chosen_model}, index_dim={index_dim}, queries={len(prompts)}")


if __name__ == "__main__":
    main()
