#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, io, json, os, random, collections, math

def read_jsonl(path):
    with io.open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def summarize(split_name, items):
    total = len(items)
    by_pack = collections.Counter([x["pack_id"] for x in items])
    by_label = collections.Counter([x["label"] for x in items])
    packs = len(by_pack)
    dist = {k: (v, round(v/max(1,total),3)) for k,v in sorted(by_label.items())}
    print(f"[{split_name}] examples={total} packs={packs} labels={dist}")

def main():
    ap = argparse.ArgumentParser("Split mcq_answer JSONL by pack_id")
    ap.add_argument("--inp", required=True, help="Path to mcq_answer.topik1.jsonl")
    ap.add_argument("--outdir", required=True, help="Output directory for splits/")
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.90, 0.05, 0.05],
                    help="Train/Val/Test ratios, e.g. 0.9 0.05 0.05")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    r_train, r_val, r_test = args.ratios
    s = r_train + r_val + r_test
    if not (abs(s - 1.0) < 1e-6):
        raise SystemExit(f"Ratios must sum to 1.0, got {args.ratios} (sum={s})")

    data = list(read_jsonl(args.inp))
    if not data:
        raise SystemExit(f"No data in {args.inp}")

    # group by pack_id
    packs = {}
    for ex in data:
        pid = ex.get("pack_id")
        if not pid:
            raise SystemExit("Missing pack_id in an example.")
        packs.setdefault(pid, []).append(ex)

    pack_keys = list(packs.keys())
    rng = random.Random(args.seed)
    rng.shuffle(pack_keys)

    nP = len(pack_keys)
    n_train = int(round(nP * r_train))
    n_val   = int(round(nP * r_val))
    # ensure sum equals nP
    n_test  = nP - n_train - n_val
    if n_test < 0:
        n_test = 0
        n_val  = nP - n_train

    train_keys = set(pack_keys[:n_train])
    val_keys   = set(pack_keys[n_train:n_train+n_val])
    test_keys  = set(pack_keys[n_train+n_val:])

    def collect(keys):
        out = []
        for k in keys:
            out.extend(packs[k])
        return out

    train = collect(train_keys)
    val   = collect(val_keys)
    test  = collect(test_keys)

    # write
    out_train = os.path.join(args.outdir, "train.jsonl")
    out_val   = os.path.join(args.outdir, "val.jsonl")
    out_test  = os.path.join(args.outdir, "test.jsonl")
    write_jsonl(out_train, train)
    write_jsonl(out_val, val)
    write_jsonl(out_test, test)

    # report
    print(f"Total packs={nP} => train={len(train_keys)}, val={len(val_keys)}, test={len(test_keys)}")
    summarize("train", train)
    summarize("val", val)
    summarize("test", test)

if __name__ == "__main__":
    main()
