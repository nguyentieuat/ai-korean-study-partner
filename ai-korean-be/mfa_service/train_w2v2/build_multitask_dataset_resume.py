#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build Multi-Task ASR dataset from MFA CSV alignments with RESUME & PARALLEL

Tasks:
- Phones (có thể tắt bằng --no-phones)
    * --phone-mode group   : gom phones liền kề cho đủ min_ms..max_ms (>= min_phones)
    * --phone-mode byword  : phones cắt theo đúng khung thời gian word (đồng bộ word)
    * --phone-mode sliding : phones cắt theo cửa sổ trượt (giống jamo-streaming)
- Word
    * clip >= 350 ms, cover full word (+ kéo nhẹ nếu ngắn)
    * targets: word + jamo(NFD) + phonemes(khớp cửa sổ)
- Jamo-streaming
    * sliding windows (win=600ms, stride=200ms mặc định)

Tính năng:
- --resume: bỏ qua CSV đã sinh tmp manifest + .done
- --workers N: xử lý song song nhiều CSV
- --phone-max-chunks-per-utt: giới hạn số clip phones mỗi CSV (0 = no limit)
- --no-phones: bỏ hoàn toàn việc tạo clip task=phones
- Chuẩn hóa Jamo về Unicode NFD
"""

import os, json, argparse, unicodedata
from pathlib import Path
from typing import List, Tuple, Dict, Set
import pandas as pd
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor, as_completed

# -------- Helpers --------
def sec_to_ms(x: float) -> int: return int(round(x * 1000))
def ms_to_samples(ms: int, sr: int) -> int: return int(round(ms * sr / 1000.0))
def clamp(a, lo, hi): return max(lo, min(hi, a))

def normalize_NFD_list(seq: List[str]) -> List[str]:
    joined = "".join(seq)
    normed = unicodedata.normalize("NFD", joined)
    return [j for j in normed if j.strip()]

def load_alignment(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    need = {"Begin","End","Label","Type"}
    miss = need - set(df.columns)
    if miss: raise ValueError(f"{csv_path.name}: missing {miss}")
    df["Begin"] = df["Begin"].astype(float)
    df["End"]   = df["End"].astype(float)
    df["Label"] = df["Label"].astype(str)
    if "Speaker" not in df.columns:
        df["Speaker"] = "corpus"
    return df

def slice_audio(wav_path: Path, start_ms: int, end_ms: int, out_path: Path):
    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    n = data.shape[0]
    s = clamp(ms_to_samples(start_ms, sr), 0, n)
    e = clamp(ms_to_samples(end_ms, sr), 0, n)
    if e <= s: return False, sr
    seg = data[s:e] if data.ndim == 1 else data[s:e, :]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), seg, sr, subtype="PCM_16")
    return True, sr

# -------- Hangul decomposition --------
_CHOSEONG  = [chr(x) for x in range(0x1100, 0x1113)]
_JUNGSEONG = [chr(x) for x in range(0x1161, 0x1176)]
_JONGSEONG = [""] + [chr(x) for x in range(0x11A8, 0x11C3)]

def decompose_hangul_to_jamo(s: str) -> List[str]:
    out = []
    for ch in s:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            syll = code - 0xAC00
            c = syll // 588
            v = (syll % 588) // 28
            t = syll % 28
            out.extend([_CHOSEONG[c], _JUNGSEONG[v]])
            if t != 0:
                out.append(_JONGSEONG[t])
        elif ch.strip():
            out.append(ch)
    return normalize_NFD_list(out)

# -------- Window queries --------
def phones_in_window(phones_df, st_ms, en_ms):
    hit = phones_df[(phones_df["Begin"]*1000 < en_ms) & (phones_df["End"]*1000 > st_ms)]
    return [(sec_to_ms(r["Begin"]), sec_to_ms(r["End"]), r["Label"].strip())
            for _, r in hit.sort_values("Begin").iterrows()]

def words_in_window(words_df, st_ms, en_ms):
    hit = words_df[(words_df["Begin"]*1000 < en_ms) & (words_df["End"]*1000 > st_ms)]
    return [(sec_to_ms(r["Begin"]), sec_to_ms(r["End"]), r["Label"].strip())
            for _, r in hit.sort_values("Begin").iterrows() if r["Label"].strip()]

# -------- Chunk builders --------
def build_chunks_by_phones_group(phones_df, min_ms=300, max_ms=700, min_phones=2, stride_phones=1):
    """
    Gộp các phones liên tiếp thành một đoạn [st,en] sao cho:
      - Độ dài trong [min_ms, max_ms] (có co bớt cuối nếu vượt)
      - Chứa ít nhất min_phones phones
    Trượt cửa sổ theo chỉ số phone (stride_phones).
    """
    out = []
    ph = phones_df.sort_values("Begin").reset_index(drop=True)
    n = len(ph)
    i = 0
    while i < n:
        st = sec_to_ms(ph.loc[i, "Begin"])
        en = sec_to_ms(ph.loc[i, "End"])
        j = i
        while j + 1 < n and (en - st) < min_ms:
            j += 1
            en = sec_to_ms(ph.loc[j, "End"])
        num_phones = j - i + 1
        dur = en - st
        if num_phones >= min_phones and dur >= min_ms:
            while dur > max_ms and j > i:
                j -= 1
                en = sec_to_ms(ph.loc[j, "End"])
                num_phones = j - i + 1
                dur = en - st
            if min_ms <= dur <= max_ms and num_phones >= min_phones:
                out.append((st, en))
        i += stride_phones
    return out

def build_phone_chunks_byword(words_df, min_ms=300):
    """Dùng đúng khung thời gian word để cắt phones (giống word)."""
    out = []
    wd = words_df.sort_values("Begin").reset_index(drop=True)
    for _, w in wd.iterrows():
        st = sec_to_ms(w["Begin"]); en = sec_to_ms(w["End"])
        if en - st < min_ms:
            en = st + min_ms
        out.append((st, en))
    return out

def build_chunks_sliding(start_ms, end_ms, win_ms=600, stride_ms=200):
    """Cửa sổ trượt thời gian (dùng cho phones-mode=sliding và jamo-streaming)."""
    out = []
    cur = start_ms
    while cur < end_ms:
        st = cur
        en = min(end_ms, cur + win_ms)
        out.append((st, en))
        cur += stride_ms
        if en == end_ms: break
    return out

def build_chunks_by_words(words_df, min_ms=350):
    out = []
    wd = words_df.sort_values("Begin").reset_index(drop=True)
    for _, w in wd.iterrows():
        st, en, lab = sec_to_ms(w["Begin"]), sec_to_ms(w["End"]), w["Label"].strip()
        if not lab: continue
        if en - st < min_ms: en = st + min_ms
        out.append((st, en, lab))
    return out

# -------- Worker: process ONE CSV -> write tmp manifest --------
def process_one_csv(args_pack):
    (csvf, out_root, corpus_dir, make_jamo,
     phone_min_ms, phone_max_ms, phone_min_phones, phone_stride_phones,
     phone_max_chunks_per_utt, phone_mode, phone_win_ms, phone_stride_ms,
     no_phones) = args_pack

    base = csvf.stem
    tmp_dir = out_root/"manifests"/"tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_manifest = tmp_dir/f"{base}.jsonl"
    done_flag = tmp_dir/f"{base}.done"

    if done_flag.exists() and tmp_manifest.exists():
        return {"phonemes":[], "jamo":[], "words":[]}

    wav = corpus_dir/f"{base}.wav"
    if not wav.exists():
        return {"phonemes":[], "jamo":[], "words":[]}

    try:
        df = load_alignment(csvf)
    except Exception:
        return {"phonemes":[], "jamo":[], "words":[]}

    ph_df = df[df.Type=="phones"].copy()
    wd_df = df[df.Type=="words"].copy()

    # clean non-speech
    nonspeech = {"spn","sil","noise","pau","ns"}
    ph_df = ph_df[~ph_df["Label"].isin(nonspeech)]
    wd_df = wd_df[~wd_df["Label"].isin(nonspeech)]

    if len(df)==0:
        return {"phonemes":[], "jamo":[], "words":[]}

    g0, g1 = sec_to_ms(df.Begin.min()), sec_to_ms(df.End.max())

    audio_words_dir  = out_root/"audio"/"words"
    audio_jamo_dir   = out_root/"audio"/"jamo"
    audio_phones_dir = out_root/"audio"/"phones"

    ph_vocab: Set[str] = set()
    ja_vocab: Set[str] = set()
    wd_vocab: Set[str] = set()

    with open(tmp_manifest, "w", encoding="utf-8") as fout:
        # -------- (tùy chọn) phones --------
        if not no_phones:
            if phone_mode == "group":
                phone_windows = build_chunks_by_phones_group(
                    ph_df,
                    min_ms=phone_min_ms, max_ms=phone_max_ms,
                    min_phones=phone_min_phones, stride_phones=phone_stride_phones
                )
            elif phone_mode == "byword":
                phone_windows = build_phone_chunks_byword(
                    wd_df, min_ms=phone_min_ms
                )
            else:  # "sliding"
                phone_windows = build_chunks_sliding(
                    g0, g1, win_ms=phone_win_ms, stride_ms=phone_stride_ms
                )

            count = 0
            for st, en in phone_windows:
                if phone_max_chunks_per_utt > 0 and count >= phone_max_chunks_per_utt:
                    break
                phs = [p[2] for p in phones_in_window(ph_df, st, en)]
                if phone_mode in ("byword", "sliding") and len(phs) < phone_min_phones:
                    continue
                if not phs:
                    continue
                ph_vocab.update(phs)
                out = audio_phones_dir/f"{base}_{st:08d}_{en:08d}.wav"
                ok, sr = slice_audio(wav, st, en, out)
                if not ok: continue
                obj={
                    "audio": str(out),
                    "sr": sr,
                    "dur_ms": en-st,
                    "task": "phones",
                    "targets": {"phonemes": phs, "jamo": None, "word": None}
                }
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                count += 1

        # -------- words (đính kèm phonemes cho co-supervise) --------
        for st,en,lab in build_chunks_by_words(wd_df):
            jams = decompose_hangul_to_jamo(lab)
            if jams: ja_vocab.update(jams)
            wd_vocab.add(lab)

            phs_for_word = [p[2] for p in phones_in_window(ph_df, st, en)]

            out = audio_words_dir/f"{base}_{st:08d}_{en:08d}.wav"
            ok, sr = slice_audio(wav, st, en, out)
            if not ok: continue
            obj={
                "audio": str(out),
                "sr": sr,
                "dur_ms": en-st,
                "task": "word",
                "targets": {"word": lab, "jamo": jams, "phonemes": phs_for_word}
            }
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

        # -------- jamo streaming --------
        if make_jamo and len(wd_df)>0:
            for st,en in build_chunks_sliding(g0,g1,win_ms=600, stride_ms=200):
                jams = []
                for _,_,lab in words_in_window(wd_df, st, en):
                    js = decompose_hangul_to_jamo(lab)
                    if js: jams.extend(js)
                if len(jams) < 2: continue
                ja_vocab.update(jams)
                out = audio_jamo_dir/f"{base}_{st:08d}_{en:08d}.wav"
                ok, sr = slice_audio(wav, st, en, out)
                if not ok: continue
                obj={
                    "audio": str(out),
                    "sr": sr,
                    "dur_ms": en-st,
                    "task": "jamo",
                    "targets": {"jamo": jams}
                }
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    done_flag.touch()
    return {"phonemes": sorted(ph_vocab), "jamo": sorted(ja_vocab), "words": sorted(wd_vocab)}

# -------- Merge tmp manifests + write vocabs --------
def merge_and_finalize(out_root: Path):
    mani_dir = out_root/"manifests"
    tmp_dir  = mani_dir/"tmp"
    final_mani = mani_dir/"multi_task.jsonl"
    ph_vocab, ja_vocab, wd_vocab = set(), set(), set()

    with open(final_mani, "w", encoding="utf-8") as fout:
        for f in sorted(tmp_dir.glob("*.jsonl")):
            with open(f, "r", encoding="utf-8") as fin:
                for line in fin:
                    obj = json.loads(line)
                    t = obj.get("task")
                    tg = obj.get("targets", {})
                    if t == "phones":
                        for p in tg.get("phonemes") or []: ph_vocab.add(p)
                    if t == "word":
                        w = tg.get("word")
                        if w: wd_vocab.add(w)
                        for j in tg.get("jamo") or []: ja_vocab.add(j)
                        for p in tg.get("phonemes") or []: ph_vocab.add(p)
                    if t == "jamo":
                        for j in tg.get("jamo") or []: ja_vocab.add(j)
                    fout.write(line)

    # write vocabs
    if ja_vocab:
        with open(mani_dir/"vocab_jamo.txt","w",encoding="utf-8") as f:
            for j in sorted(ja_vocab): f.write(j+"\n")
    if ph_vocab:
        with open(mani_dir/"vocab_phoneme.txt","w",encoding="utf-8") as f:
            for p in sorted(ph_vocab): f.write(p+"\n")
    if wd_vocab:
        with open(mani_dir/"vocab_word.json","w",encoding="utf-8") as f:
            json.dump({w:i for i,w in enumerate(sorted(wd_vocab))}, f, ensure_ascii=False, indent=2)

# -------- Main --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-align", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--make-jamo-streaming", action="store_true")
    ap.add_argument("--workers", type=int, default=1, help="Số tiến trình xử lý song song")
    ap.add_argument("--resume", action="store_true", help="Bỏ qua CSV đã có tmp .jsonl+.done")
    ap.add_argument("--no-phones", action="store_true", help="Bỏ qua hoàn toàn việc tạo clip task=phones")

    # phones params (vẫn giữ, nhưng sẽ bị bỏ qua nếu --no-phones)
    ap.add_argument("--phone-mode", choices=["group","byword","sliding"], default="byword",
                    help="Cách cắt phones: group (gom), byword (theo từ), sliding (cửa sổ trượt)")
    ap.add_argument("--phone-min-ms", type=int, default=300)
    ap.add_argument("--phone-max-ms", type=int, default=700)
    ap.add_argument("--phone-min-phones", type=int, default=2)
    ap.add_argument("--phone-stride-phones", type=int, default=1)
    ap.add_argument("--phone-win-ms", type=int, default=600, help="Chỉ dùng cho phone-mode=sliding")
    ap.add_argument("--phone-stride-ms", type=int, default=200, help="Chỉ dùng cho phone-mode=sliding")
    ap.add_argument("--phone-max-chunks-per-utt", type=int, default=0,
                    help="0 = không giới hạn; >0 = tối đa N chunk phones mỗi CSV")

    args = ap.parse_args()

    out_root = Path(args.out_root)
    if not args.no_phones:
        (out_root/"audio"/"phones").mkdir(parents=True, exist_ok=True)
    (out_root/"audio"/"words").mkdir(parents=True, exist_ok=True)
    if args.make_jamo_streaming:
        (out_root/"audio"/"jamo").mkdir(parents=True, exist_ok=True)
    (out_root/"manifests"/"tmp").mkdir(parents=True, exist_ok=True)

    csvs = sorted(Path(args.out_align).glob("*.csv"))
    if not csvs:
        print("No CSV found."); return

    # filter for resume
    if args.resume:
        filtered = []
        for c in csvs:
            base = c.stem
            tmp = out_root/"manifests"/"tmp"/f"{base}.jsonl"
            done = out_root/"manifests"/"tmp"/f"{base}.done"
            if tmp.exists() and done.exists():
                continue
            filtered.append(c)
        csvs = filtered

    if not csvs:
        print("Nothing to do (resume). Merging…")
        merge_and_finalize(out_root)
        print("Done (resume-merge)."); return

    tasks = [
        (c, out_root, Path(args.corpus), args.make_jamo_streaming,
         args.phone_min_ms, args.phone_max_ms, args.phone_min_phones, args.phone_stride_phones,
         args.phone_max_chunks_per_utt, args.phone_mode, args.phone_win_ms, args.phone_stride_ms,
         args.no_phones)
        for c in csvs
    ]

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(process_one_csv, t) for t in tasks]
            for _ in as_completed(futures):
                pass
    else:
        for t in tasks:
            process_one_csv(t)

    merge_and_finalize(out_root)
    print("All done. Manifest & vocabs ready.")

if __name__ == "__main__":
    main()
