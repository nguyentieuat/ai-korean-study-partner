#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Merge toàn bộ JSONL trong một thư mục, khử trùng lặp, rồi tách train/dev/test
theo tỉ lệ cho trước, có cân bằng theo (topic|question_no|answer|speaker).
"""

import os, json, argparse, glob, random, math
from collections import defaultdict, Counter

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line=line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                print(f"[WARN] {path}:{ln} parse error: {e}")

def norm_text(s: str) -> str:
    return " ".join((s or "").strip().split())

def make_key(obj):
    # dialogue[0]
    dlg = obj.get("dialogue", [{}])
    d0 = dlg[0] if dlg else {}
    spk = d0.get("speaker","")
    txt = norm_text(d0.get("text",""))
    opts = obj.get("options", {})
    # key dựa vào nội dung cốt lõi để bắt trùng
    key = (
        obj.get("question_no"),
        spk, txt,
        norm_text(opts.get("A","")),
        norm_text(opts.get("B","")),
        norm_text(opts.get("C","")),
        norm_text(opts.get("D","")),
        obj.get("answer"),
    )
    return key

def load_all(src_dir, include_glob="*.jsonl"):
    files = sorted(glob.glob(os.path.join(src_dir, include_glob)))
    print(f"[INFO] found {len(files)} files")
    data = []
    seen = set()
    for fp in files:
        for obj in read_jsonl(fp):
            # mặc định gán topic nếu thiếu
            obj.setdefault("topic", "misc")
            # chuẩn hoá speaker
            try:
                spk = obj["dialogue"][0].get("speaker","")
                if spk not in ("남자","여자"):
                    obj["dialogue"][0]["speaker"] = "남자"  # fallback
            except Exception:
                obj["dialogue"] = [{"speaker":"남자","text":""}]
            # khóa dedupe
            k = make_key(obj)
            if k in seen:
                continue
            seen.add(k)
            data.append(obj)
    print(f"[INFO] total after dedupe: {len(data)}")
    return data

def label_of(obj):
    d0 = obj["dialogue"][0]
    return f"{obj.get('topic','misc')}|{obj.get('question_no',5)}|{obj.get('answer','A')}|{d0.get('speaker','남자')}"

def stratified_split(data, train_ratio=0.9, dev_ratio=0.05, test_ratio=0.05, seed=42):
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6
    random.seed(seed)
    # gom theo nhãn
    buckets = defaultdict(list)
    for i, obj in enumerate(data):
        buckets[label_of(obj)].append(obj)
    # quota theo nhãn
    dev, test, train = [], [], []
    for lab, items in buckets.items():
        random.shuffle(items)
        n = len(items)
        n_dev  = int(round(n * dev_ratio))
        n_test = int(round(n * test_ratio))
        # giữ tổng không vượt n
        if n_dev + n_test > n:
            n_test = max(0, n - n_dev)
        dev.extend(items[:n_dev])
        test.extend(items[n_dev:n_dev+n_test])
        train.extend(items[n_dev+n_test:])
    random.shuffle(train); random.shuffle(dev); random.shuffle(test)
    return train, dev, test

def write_jsonl(path, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def inject_file(into_list, path):
    if not path: return
    for obj in read_jsonl(path):
        into_list.append(obj)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True, help=r"Thư mục chứa các .jsonl (VD: \data\topik1\listen\5-6\dataset)")
    ap.add_argument("--out_dir", required=True, help=r"Thư mục ghi các file splits")
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.9, 0.05, 0.05], help="train dev test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reserve_dev", type=str, default="", help="(tuỳ chọn) file jsonl để thêm vào dev (VD: topik_I_5_6.jsonl)")
    args = ap.parse_args()

    train_r, dev_r, test_r = args.ratios
    data = load_all(args.src_dir, "*.jsonl")
    # stratified split
    train, dev, test = stratified_split(data, train_r, dev_r, test_r, seed=args.seed)

    # (tuỳ chọn) thêm file "thật" vào dev
    if args.reserve_dev:
        inject_file(dev, args.reserve_dev)

    print("[INFO] sizes: train/dev/test =", len(train), len(dev), len(test))

    # ghi file
    write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train)
    write_jsonl(os.path.join(args.out_dir, "dev.jsonl"),   dev)
    write_jsonl(os.path.join(args.out_dir, "test.jsonl"),  test)

    # thống kê
    def stats(name, items):
        a=Counter([x["answer"] for x in items])
        q=Counter([x["question_no"] for x in items])
        s=Counter([x["dialogue"][0]["speaker"] for x in items])
        t=Counter([x.get("topic","misc") for x in items])
        print(f"[{name}] answers={dict(a)}  qno={dict(q)}  speaker={dict(s)}  topics(top5)={t.most_common(5)}")

    stats("TRAIN", train)
    stats("DEV",   dev)
    stats("TEST",  test)

if __name__ == "__main__":
    main()
