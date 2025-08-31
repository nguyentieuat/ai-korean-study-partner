#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, re, math
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Optional RAG
try:
    import faiss  # type: ignore
    import numpy as np
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False

try:
    from sentence_transformers import SentenceTransformer
    HAVE_ST = True
except Exception:
    HAVE_ST = False


def read_jsonl(p: str) -> List[Dict[str, Any]]:
    data = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                data.append(json.loads(s))
    return data


def normalize_label(s: str) -> str:
    # Chuẩn hoá output về 'A'/'B'/'C'/'D'
    s = s.strip()
    # map số -> chữ
    num_map = {"1":"A","2":"B","3":"C","4":"D","①":"A","②":"B","③":"C","④":"D"}
    if len(s)==1 and s.upper() in ["A","B","C","D"]:
        return s.upper()
    if s in num_map: return num_map[s]
    # regex tìm chữ cái đầu tiên thuộc [ABCD]
    m = re.search(r"\b([ABCD])\b", s.upper())
    if m: return m.group(1)
    # regex số đầu tiên thuộc 1..4
    m = re.search(r"\b([1-4])\b", s)
    if m: return num_map[m.group(1)]
    # ký tự Hàn/Việt có thể dính kèm, fallback lấy chữ cái đầu
    for ch in s.upper():
        if ch in "ABCD":
            return ch
    return ""


def build_prompt(base_prompt: str, rag_texts: List[str], add_rule: bool=True) -> str:
    out = base_prompt
    if rag_texts:
        out += "\n\n참고 지문:\n" + "\n---\n".join(rag_texts)
    if add_rule and "정답의 글자만 출력하세요" not in out:
        out += "\n\n규칙: 정답의 글자만 출력하세요."
    return out


def load_index(indexdir: str):
    # Đọc FAISS + texts + meta
    meta = json.load(open(Path(indexdir)/"meta.json", "r", encoding="utf-8"))
    texts_path = Path(indexdir)/"texts.jsonl"
    texts = [json.loads(l)["text"] for l in open(texts_path, "r", encoding="utf-8")]
    # load faiss index
    index_files = list(Path(indexdir).glob("index*.faiss"))
    if not index_files:
        raise FileNotFoundError("Không tìm thấy file index*.faiss trong indexdir.")
    index = faiss.read_index(str(index_files[0]))
    return meta, texts, index


def encode_queries(queries: List[str], enc_name: str):
    # e5 cần prefix 'query: '
    use_e5 = "e5" in enc_name.lower()
    if HAVE_ST:
        model = SentenceTransformer(enc_name, device="cpu")
        q = [("query: " + q) if use_e5 else q for q in queries]
        X = model.encode(q, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        return np.asarray(X, dtype="float32")
    else:
        raise RuntimeError("Cần sentence-transformers để dùng RAG. Cài: pip install sentence-transformers")


def search_topk(index, Xq, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    scores, ids = index.search(Xq, topk)
    return scores, ids


def get_rag_chunks(query: str, index_bundle, topk: int) -> List[str]:
    meta, texts, index = index_bundle
    Xq = encode_queries([query], meta.get("model_name", ""))
    scores, ids = search_topk(index, Xq, topk)
    out = []
    for j in ids[0]:
        if j >= 0 and j < len(texts):
            # cắt gọn đoạn quá dài
            t = texts[j]
            out.append(t if len(t)<=700 else t[:700]+" ...")
    return out


def run_eval(model_id: str, lora_path: str, cls_path: str,
             indexdir: str, topk: int, device: str,
             limit: int, max_new_tokens: int, temperature: float) -> Dict[str, Any]:

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Model
    if device == "auto":
        device_map = "auto"
    elif device.startswith("cuda"):
        device_map = device
    else:
        device_map = {"": "cpu"}

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map=device_map, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    # (Optional) LoRA
    if lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)

    # (Optional) RAG index
    index_bundle = None
    if indexdir:
        if not HAVE_FAISS:
            raise RuntimeError("Cần `faiss-cpu` để RAG. Cài: pip install faiss-cpu")
        index_bundle = load_index(indexdir)

    data = read_jsonl(cls_path)
    if limit and limit > 0:
        data = data[:limit]

    total = 0
    correct = 0
    details = []

    model.eval()
    torch.set_grad_enabled(False)

    for i, ex in enumerate(data):
        # chấp nhận nhiều schema nhãn
        gold = ex.get("label") or ex.get("answer") or ex.get("gold") or ex.get("ref") or ""
        gold = normalize_label(str(gold))
        prompt = ex.get("prompt") or ex.get("input") or ex.get("question") or ""

        # thêm RAG nếu có
        rag_txts = []
        if index_bundle and topk > 0:
            rag_txts = get_rag_chunks(prompt, index_bundle, topk)

        full_prompt = build_prompt(prompt, rag_txts, add_rule=True)

        # chat template
        messages = [{"role":"user", "content": full_prompt}]
        inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

        gen = model.generate(
            **inputs,
            do_sample=(temperature > 0.0),
            temperature=max(1e-6, temperature),
            top_p=1.0,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        out_text = tok.decode(gen[0], skip_special_tokens=True)
        # Lấy phần "assistant" cuối cùng
        cand = out_text.split(messages[0]["content"])[-1].strip()
        pred = normalize_label(cand)

        ok = (pred == gold) if gold else False
        total += 1
        correct += int(ok)
        details.append({"i": i, "gold": gold, "pred": pred, "ok": ok})

        if (i+1) % 20 == 0:
            acc = correct / total * 100.0
            print(f"[{i+1}/{len(data)}] acc={acc:.2f}%")

    acc = correct / max(1,total) * 100.0
    return {"acc": acc, "n": total, "correct": correct, "details": details}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id hoặc đường dẫn local")
    ap.add_argument("--lora", default="", help="đường dẫn adapter LoRA (tuỳ chọn)")
    ap.add_argument("--cls", required=True, help="file *.classification.jsonl")
    ap.add_argument("--indexdir", default="", help="thư mục korquad_index để RAG (tuỳ chọn)")
    ap.add_argument("--topk", type=int, default=0, help="số đoạn RAG (0 = tắt RAG)")
    ap.add_argument("--device", default="auto", help='auto|cpu|cuda|cuda:0')
    ap.add_argument("--limit", type=int, default=0, help="chỉ đánh giá N mẫu đầu (0 = tất cả)")
    ap.add_argument("--max_new_tokens", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--save_pred", default="", help="nếu set, ghi chi tiết dự đoán ra JSONL")
    args = ap.parse_args()

    res = run_eval(
        model_id=args.model,
        lora_path=args.lora,
        cls_path=args.cls,
        indexdir=args.indexdir,
        topk=args.topk,
        device=args.device,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(f"[RESULT] acc={res['acc']:.2f}% ({res['correct']}/{res['n']})")

    if args.save_pred:
        with open(args.save_pred, "w", encoding="utf-8") as f:
            for d in res["details"]:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
