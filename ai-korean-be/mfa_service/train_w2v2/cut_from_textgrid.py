#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cut_from_textgrid.py
Cut short audio clips using MFA TextGrid boundaries (phones/words),
with options to pack to a target duration and snap cut boundaries to word edges.

Requires:
  pip install pydub textgrid tqdm

Usage examples:

# 1) Export every non-silence phone as one clip (very short)
python cut_from_textgrid.py \
  --root "E:/USE/My_project/source audio/aligned" \
  --out  "E:/USE/My_project/source audio/output_textgrid" \
  --tier phones --mode per_interval \
  --sil_labels sil,sp,spn,nsn,<unk> \
  --min_ms 120 --max_ms 2000 --pad_ms 120

# 2) Pack adjacent phones up to ~500 ms but DON'T cross word boundaries
python cut_from_textgrid.py \
  --root "E:/USE/My_project/source audio/aligned" \
  --out  "E:/USE/My_project/source audio/output_textgrid_phones" \
  --tier phones --mode pack_to_target --target_ms 500 \
  --respect_words true --snap_to_words true --snap_tolerance_ms 60 \
  --sil_labels sil,sp,spn,nsn,<unk> \
  --min_ms 200 --max_ms 1500 --pad_ms 160

# 3) Pack whole words to 400–700 ms (never cut inside a word)
python cut_from_textgrid.py \
  --root "E:/USE/My_project/source audio/aligned" \
  --out  "E:/USE/My_project/source audio/chunks_words_400_700" \
  --tier words --mode pack_to_target --target_ms 550 \
  --min_ms 400 --max_ms 700 \
  --sil_labels sil,sp,spn,nsn,<unk> \
  --pad_ms 150

Notes:
- Put WAV and TextGrid with the same stem in the same folders:
    foo.wav  +  foo.TextGrid
- --tier can be "phones", "phone", "segments", "words" etc., depending on your MFA recipe.
- If --respect_words true and a "words" tier exists, segments will not cross word boundaries.
- --snap_to_words adjusts boundaries (±tolerance) to the nearest word edges to avoid mid-word cuts.
"""

import os, csv, argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
from pydub import AudioSegment
from pydub import effects as pd_effects
import textgrid

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
TG_EXTS = {".TextGrid", ".textgrid"}

def find_pairs(root: Path) -> List[Tuple[Path, Path]]:
    pairs = []
    for wav in root.rglob("*"):
        if not wav.is_file() or wav.suffix.lower() not in AUDIO_EXTS:
            continue
        stem = wav.with_suffix("")
        tg = None
        for tgext in TG_EXTS:
            cand = stem.with_suffix(tgext)
            if cand.exists():
                tg = cand; break
        if tg is not None:
            pairs.append((wav, tg))
    return pairs

def ensure_mono16k(seg: AudioSegment) -> AudioSegment:
    if seg.channels != 1: seg = seg.set_channels(1)
    if seg.frame_rate != 16000: seg = seg.set_frame_rate(16000)
    if seg.sample_width != 2: seg = seg.set_sample_width(2)  # PCM16
    return seg

def sec_to_ms(x: float) -> int:
    return max(0, int(round(x * 1000.0)))

def export_clip(src_path: Path, seg: AudioSegment, s_ms: int, e_ms: int, out_root: Path,
                pad_ms: int = 0, ensure_min_ms: int = 0) -> Tuple[Path, int, int]:
    L = len(seg)
    # padding + ensure min
    s = max(0, s_ms - pad_ms)
    e = min(L, e_ms + pad_ms)
    need = ensure_min_ms - (e - s)
    if ensure_min_ms and need > 0:
        add_left = min(need // 2, s)
        add_right = min(need - add_left, L - e)
        s -= add_left; e += add_right
    rel = src_path.relative_to(ROOT_DIR).with_suffix("")
    out_dir = out_root / rel
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{src_path.stem}_{s:06d}-{e:06d}.wav"
    clip = seg[s:e]
    clip.export(out_path, format="wav", parameters=["-acodec", "pcm_s16le"])
    return out_path, s, e

def pick_tier(tg: textgrid.TextGrid, want: str) -> Optional[textgrid.IntervalTier]:
    # Case-insensitive contains
    want_low = want.lower().strip()
    for tier in tg.tiers:
        name = (tier.name or "").lower()
        if want_low == name or want_low in name:
            if isinstance(tier, textgrid.IntervalTier):
                return tier
    # fallback: first interval tier
    for tier in tg.tiers:
        if isinstance(tier, textgrid.IntervalTier):
            return tier
    return None

def load_words_tier(tg: textgrid.TextGrid) -> Optional[textgrid.IntervalTier]:
    # Try tier named "words" or "word"
    for name in ["words", "word"]:
        for tier in tg.tiers:
            if isinstance(tier, textgrid.IntervalTier) and (tier.name or "").lower() == name:
                return tier
    # else best-effort by substring
    for tier in tg.tiers:
        nm = (tier.name or "").lower()
        if isinstance(tier, textgrid.IntervalTier) and "word" in nm:
            return tier
    return None

def intervals_from_tier(tier: textgrid.IntervalTier) -> List[Tuple[int,int,str]]:
    out = []
    for iv in tier.intervals:
        s = sec_to_ms(iv.minTime); e = sec_to_ms(iv.maxTime)
        lab = (iv.mark or "").strip()
        if e > s:
            out.append((s, e, lab))
    return out

# ---------- Helpers for word-aware cutting ----------
def compute_word_bounds(all_word_intervals: List[Tuple[int,int,str]]) -> List[Tuple[int,int]]:
    """Return (start, end) for ALL word intervals (including silence tokens), to know true word edges."""
    return [(s, e) for (s, e, _) in all_word_intervals]

def compute_non_silence_words(all_word_intervals: List[Tuple[int,int,str]], sil_set: set) -> List[Tuple[int,int]]:
    """Return only non-silence word intervals to detect mid-word cuts."""
    return [(s, e) for (s, e, lab) in all_word_intervals if (lab != "" and lab.lower() not in sil_set)]

def snap_ranges_to_word_boundaries(ranges: List[Tuple[int,int]],
                                   all_word_intervals: List[Tuple[int,int,str]],
                                   sil_set: set,
                                   tolerance_ms: int,
                                   audio_len_ms: int) -> List[Tuple[int,int]]:
    """If a boundary falls inside a non-silence word interval, move it to nearest word edge within tolerance."""
    if tolerance_ms <= 0 or not ranges or not all_word_intervals:
        return ranges

    non_sil_words = compute_non_silence_words(all_word_intervals, sil_set)
    snapped = []

    def nearest_edge_within_tolerance(t: int) -> Optional[int]:
        for ws, we in non_sil_words:
            if ws < t < we:  # inside a word
                # pick nearest edge
                left_dist = abs(t - ws)
                right_dist = abs(we - t)
                edge = ws if left_dist <= right_dist else we
                if min(left_dist, right_dist) <= tolerance_ms:
                    return edge
                else:
                    return None
        return None

    for (s, e) in ranges:
        ns = nearest_edge_within_tolerance(s)
        ne = nearest_edge_within_tolerance(e)
        s2 = max(0, ns if ns is not None else s)
        e2 = min(audio_len_ms, ne if ne is not None else e)
        if e2 <= s2:
            # avoid inversion: keep original if snapping collapses
            s2, e2 = s, e
        snapped.append((s2, e2))
    return snapped

# ---------- Packing / Per-interval ----------
def pack_to_target(intervals: List[Tuple[int,int,str]], sil_set: set,
                   target_ms: int, min_ms: int, max_ms: int,
                   respect_words: bool, word_bounds: Optional[List[Tuple[int,int]]] = None) -> List[Tuple[int,int]]:
    """
    Greedy packing of adjacent non-silence intervals up to ~target_ms.
    Will not cross word boundaries if respect_words and word_bounds provided.
    """
    packs = []
    cur_s = None
    cur_e = None
    i = 0

    def within_word_bounds(s: int, e: int) -> bool:
        if not respect_words or not word_bounds: return True
        # segment must be fully inside a single word interval
        for ws, we in word_bounds:
            if s >= ws and e <= we: return True
        return False

    while i < len(intervals):
        s, e, lab = intervals[i]
        if lab == "" or lab.lower() in sil_set:
            if cur_s is not None:
                packs.append((cur_s, cur_e))
                cur_s = cur_e = None
            i += 1; continue

        if cur_s is None:
            cur_s, cur_e = s, e
        else:
            new_e = e
            if not within_word_bounds(cur_s, new_e):
                packs.append((cur_s, cur_e))
                cur_s, cur_e = s, e
            else:
                cur_e = new_e

        if cur_s is not None and (cur_e - cur_s) >= target_ms:
            packs.append((cur_s, cur_e))
            cur_s = cur_e = None
        i += 1

    if cur_s is not None:
        packs.append((cur_s, cur_e))

    # Enforce min & max by splitting or dropping
    out = []
    for s, e in packs:
        dur = e - s
        if dur < min_ms:
            continue
        if dur <= max_ms:
            out.append((s, e)); continue
        cur = s
        while e - cur > max_ms:
            out.append((cur, cur + max_ms))
            cur += max_ms
        if e - cur >= min_ms:
            out.append((cur, e))
    return out

def per_interval(intervals: List[Tuple[int,int,str]], sil_set: set,
                 min_ms: int, max_ms: int) -> List[Tuple[int,int]]:
    out = []
    for s, e, lab in intervals:
        if lab == "" or lab.lower() in sil_set:
            continue
        dur = e - s
        if dur < min_ms:
            continue
        if dur <= max_ms:
            out.append((s, e))
        else:
            cur = s
            while e - cur > max_ms:
                out.append((cur, cur + max_ms))
                cur += max_ms
            if e - cur >= min_ms:
                out.append((cur, e))
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cut clips based on MFA TextGrid boundaries")
    ap.add_argument("--root", required=True, help="Root folder containing WAV+TextGrid pairs")
    ap.add_argument("--out", required=True, help="Output folder")
    ap.add_argument("--tier", default="phones", help="Tier name to use (e.g., phones, segments, words)")
    ap.add_argument("--mode", default="pack_to_target", choices=["per_interval","pack_to_target"])
    ap.add_argument("--target_ms", type=int, default=500, help="Target duration per segment (only for pack_to_target)")
    ap.add_argument("--min_ms", type=int, default=200)
    ap.add_argument("--max_ms", type=int, default=1500)
    ap.add_argument("--pad_ms", type=int, default=160)
    ap.add_argument("--ensure_min_ms", type=int, default=0)
    ap.add_argument("--sil_labels", default="sil,sp,spn,nsn,<unk>,SIL,SP,SILENCE,NSN,SPN,UNK",
                    help="Comma-separated labels treated as silence")
    ap.add_argument("--respect_words", type=str, default="true", help="true/false: avoid crossing word boundaries if words tier exists")
    ap.add_argument("--snap_to_words", action="store_true", help="Snap cut boundaries to nearest word edge if inside a word")
    ap.add_argument("--snap_tolerance_ms", type=int, default=60, help="Max distance allowed when snapping to a word boundary")
    ap.add_argument("--pre_normalize", action="store_true", help="Normalize audio before cutting (does not affect alignment)")

    args = ap.parse_args()
    global ROOT_DIR
    ROOT_DIR = Path(args.root)
    OUT_ROOT = Path(args.out)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    sil_set = {x.strip().lower() for x in args.sil_labels.split(",") if x.strip()}
    respect_words = str(args.respect_words).lower() in ("1","true","yes","y")

    pairs = find_pairs(ROOT_DIR)
    if not pairs:
        print("No WAV+TextGrid pairs found under", ROOT_DIR)
        raise SystemExit

    csv_path = OUT_ROOT / "manifest_textgrid.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_path","textgrid","tier","start_ms","end_ms","slice_path","duration_ms","mode","pad_ms","snapped"])

    total = 0
    for wav_path, tg_path in tqdm(pairs, desc="Cutting", unit="pair"):
        try:
            seg = AudioSegment.from_file(wav_path)
            seg = ensure_mono16k(seg)
            if args.pre_normalize:
                seg = pd_effects.normalize(seg)

            tg = textgrid.TextGrid.fromFile(str(tg_path))
            tier = pick_tier(tg, args.tier)
            if tier is None:
                print(f"[WARN] {wav_path.name}: tier '{args.tier}' not found, skipping.")
                continue
            intervals = intervals_from_tier(tier)

            # optional words tier
            all_word_intervals = []
            word_bounds = None
            wt = load_words_tier(tg)
            if wt:
                all_word_intervals = intervals_from_tier(wt)  # includes silence tokens
                word_bounds = [(s, e) for (s, e, _) in all_word_intervals]

            if args.mode == "per_interval":
                ranges = per_interval(intervals, sil_set, args.min_ms, args.max_ms)
            else:
                ranges = pack_to_target(intervals, sil_set, args.target_ms, args.min_ms, args.max_ms,
                                        respect_words=respect_words, word_bounds=word_bounds)

            if not ranges:
                continue

            snapped_flag = False
            if args.snap_to_words and all_word_intervals:
                old_ranges = ranges[:]
                ranges = snap_ranges_to_word_boundaries(ranges, all_word_intervals, sil_set,
                                                        tolerance_ms=args.snap_tolerance_ms,
                                                        audio_len_ms=len(seg))
                snapped_flag = (ranges != old_ranges)

            with csv_path.open("a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                for s, e in ranges:
                    out_wav, s2, e2 = export_clip(wav_path, seg, s, e, OUT_ROOT,
                                                  pad_ms=args.pad_ms, ensure_min_ms=args.ensure_min_ms)
                    w.writerow([str(wav_path), str(tg_path), tier.name, s2, e2, str(out_wav), e2 - s2, args.mode, args.pad_ms, int(snapped_flag)])
                    total += 1

        except Exception as ex:
            print(f"[WARN] {wav_path.name}: {ex}")

    print(f"DONE. total clips: {total}")
    print("CSV :", csv_path.resolve())
    print("OUT :", OUT_ROOT.resolve())
