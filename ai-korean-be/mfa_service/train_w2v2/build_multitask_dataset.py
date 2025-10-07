#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build Multi-Task ASR dataset from MFA CSV alignments:
- Phone-clips (300–700 ms)                     -> task="phones"
- Word-clips  (350–1200 ms, cover full word)   -> task="word"
- Jamo-streaming clips (sliding windows)       -> task="jamo"
- Normalize all Jamo to NFD form (Unicode decomposed)
"""

import os, csv, json, math, argparse, unicodedata
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

# ==== helpers ====
def sec_to_ms(x: float) -> int: return int(round(x * 1000))
def ms_to_samples(ms: int, sr: int) -> int: return int(round(ms * sr / 1000.0))
def clamp(a, lo, hi): return max(lo, min(hi, a))

def normalize_NFD_list(seq: List[str]) -> List[str]:
    """Normalize a list of Jamo chars to NFD."""
    joined = "".join(seq)
    normed = unicodedata.normalize("NFD", joined)
    return [j for j in normed if j.strip()]

def load_alignment(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    needed = {"Begin", "End", "Label", "Type"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name}: missing {missing}")
    df["Begin"] = df["Begin"].astype(float)
    df["End"] = df["End"].astype(float)
    df["Label"] = df["Label"].astype(str)
    if "Speaker" not in df.columns:
        df["Speaker"] = "corpus"
    return df

def slice_audio(wav_path: Path, start_ms: int, end_ms: int, out_path: Path) -> Tuple[bool, int]:
    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    n = data.shape[0] if data.ndim == 1 else data.shape[0]
    s = clamp(ms_to_samples(start_ms, sr), 0, n)
    e = clamp(ms_to_samples(end_ms, sr), 0, n)
    if e <= s: return False, sr
    seg = data[s:e] if data.ndim == 1 else data[s:e, :]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), seg, sr, subtype="PCM_16")
    return True, sr

# ==== Hangul Jamo ====
_CHOSEONG = [chr(x) for x in range(0x1100, 0x1113)]
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

# ==== window queries ====
def phones_in_window(phones_df, st_ms, en_ms):
    hit = phones_df[(phones_df["Begin"]*1000 < en_ms) & (phones_df["End"]*1000 > st_ms)]
    return [(sec_to_ms(r["Begin"]), sec_to_ms(r["End"]), r["Label"].strip())
            for _, r in hit.sort_values("Begin").iterrows()]

def words_in_window(words_df, st_ms, en_ms):
    hit = words_df[(words_df["Begin"]*1000 < en_ms) & (words_df["End"]*1000 > st_ms)]
    return [(sec_to_ms(r["Begin"]), sec_to_ms(r["End"]), r["Label"].strip())
            for _, r in hit.sort_values("Begin").iterrows() if r["Label"].strip()]

# ==== phone & word chunk builders ====
def build_chunks_by_phones(phones_df, min_ms, max_ms):
    chunks = []
    ph = phones_df.sort_values("Begin").reset_index(drop=True)
    i = 0
    while i < len(ph):
        st = sec_to_ms(ph.loc[i,"Begin"]); en = sec_to_ms(ph.loc[i,"End"])
        if en-st >= min_ms:
            chunks.append((st,en))
        i += 1
    return chunks

def build_chunks_by_words(words_df, min_ms, max_ms):
    out = []
    for _, w in words_df.iterrows():
        st, en, lab = sec_to_ms(w["Begin"]), sec_to_ms(w["End"]), w["Label"].strip()
        if not lab: continue
        dur = en - st
        if dur < min_ms: en = st + min_ms
        if dur > max_ms: max_ms = dur
        out.append((st, en, lab))
    return out

# ==== sliding jamo windows ====
def build_jamo_windows(start, end, win_ms, stride):
    out = []
    cur = start
    while cur < end:
        st, en = cur, min(end, cur + win_ms)
        out.append((st, en))
        cur += stride
        if en == end: break
    return out

def jamo_from_words(words_df, st, en):
    seq = []
    for _,_,lab in words_in_window(words_df, st, en):
        seq.extend(decompose_hangul_to_jamo(lab))
    return seq

# ==== main ====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-align", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--make-jamo-streaming", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    (out_root/"audio"/"phones").mkdir(parents=True, exist_ok=True)
    (out_root/"audio"/"words").mkdir(parents=True, exist_ok=True)
    if args.make_jamo_streaming:
        (out_root/"audio"/"jamo").mkdir(parents=True, exist_ok=True)
    mani_dir = out_root/"manifests"; mani_dir.mkdir(exist_ok=True, parents=True)
    fout = open(mani_dir/"multi_task.jsonl","w",encoding="utf-8")

    jamo_set=set(); ph_set=set(); word_to_id={}
    for csvf in tqdm(sorted(Path(args.out_align).glob("*.csv")), desc="Building"):
        base = csvf.stem
        wav = Path(args.corpus)/f"{base}.wav"
        if not wav.exists(): continue
        df = load_alignment(csvf)
        spk = df.get("Speaker","corpus")
        ph_df = df[df.Type=="phones"]; wd_df=df[df.Type=="words"]

        # --- clean non-speech tokens ---
        nonspeech = {"spn", "sil", "noise", "pau", "ns"}
        ph_df = ph_df[~ph_df["Label"].isin(nonspeech)].copy()
        wd_df = wd_df[~wd_df["Label"].isin(nonspeech)].copy()
        
        g0,g1 = sec_to_ms(df.Begin.min()), sec_to_ms(df.End.max())

        # phones
        for st,en in build_chunks_by_phones(ph_df,300,700):
            phs=[p[2] for p in phones_in_window(ph_df,st,en)]
            ph_set.update(phs)
            out=out_root/"audio"/"phones"/f"{base}_{st:08d}_{en:08d}.wav"
            ok,sr=slice_audio(wav,st,en,out)
            if not ok: continue
            obj={
                "audio":str(out),"sr":sr,"dur_ms":en-st,"task":"phones",
                "targets":{"phonemes":phs,"jamo":None,"word":None,"word_id":None}
            }
            fout.write(json.dumps(obj,ensure_ascii=False)+"\n")

        # words
        for st,en,lab in build_chunks_by_words(wd_df,350,1200):
            jams=decompose_hangul_to_jamo(lab)
            jamo_set.update(jams)
            wid=word_to_id.setdefault(lab,len(word_to_id))
            out=out_root/"audio"/"words"/f"{base}_{st:08d}_{en:08d}.wav"
            ok,sr=slice_audio(wav,st,en,out)
            if not ok: continue
            obj={
                "audio":str(out),"sr":sr,"dur_ms":en-st,"task":"word",
                "targets":{"word":lab,"word_id":wid,"jamo":jams}
            }
            fout.write(json.dumps(obj,ensure_ascii=False)+"\n")

        # jamo streaming
        if args.make_jamo_streaming:
            for st,en in build_jamo_windows(g0,g1,600,200):
                jams=jamo_from_words(wd_df,st,en)
                if len(jams)<2: continue
                out=out_root/"audio"/"jamo"/f"{base}_{st:08d}_{en:08d}.wav"
                ok,sr=slice_audio(wav,st,en,out)
                if not ok: continue
                jamo_set.update(jams)
                obj={
                    "audio":str(out),"sr":sr,"dur_ms":en-st,"task":"jamo",
                    "targets":{"jamo":jams}
                }
                fout.write(json.dumps(obj,ensure_ascii=False)+"\n")

    fout.close()
    with open(mani_dir/"vocab_jamo.txt","w",encoding="utf-8") as f:
        for j in sorted(jamo_set): f.write(j+"\n")
    with open(mani_dir/"vocab_phoneme.txt","w",encoding="utf-8") as f:
        for j in sorted(ph_set): f.write(j+"\n")
    with open(mani_dir/"vocab_word.json","w",encoding="utf-8") as f:
        json.dump(word_to_id,f,ensure_ascii=False,indent=2)
    print("Done; normalized Jamo to NFD ✔")

if __name__=="__main__":
    main()
