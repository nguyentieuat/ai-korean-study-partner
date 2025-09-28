#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
split_micro_vad_no_midword.py
Cut short clips but avoid cutting inside words/syllables by:
  - Smoothing VAD regions (merge small gaps)
  - Snapping cut boundaries to local energy minima (±snap_ms)
  - Enforcing a minimum speech length

Supports pydub "silence" VAD and WebRTC VAD, with optional pre-filtering.

Usage (recommended to avoid mid-word cuts):
  python split_micro_vad_no_midword.py \
    --root "E:/USE/My_project/source audio/audio train w2v2_16k" \
    --out  "E:/USE/My_project/source audio/output_no_midword" \
    --vad webrtc --vad_aggressiveness 2 --vad_frame_ms 30 \
    --silence_keep_ms 200 --silence_join_ms 260 \
    --merge_gap_ms 240 --snap_ms 60 --min_speech_ms 500 \
    --min_clip_ms 400 --max_ms 15000 \
    --force_window_ms 0 \
    --pad_ms 180 \
    --pre_highpass 120 --pre_lowpass 8000 --pre_normalize
"""

import os, csv, json, argparse, re
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
from pydub import AudioSegment
from pydub import silence as pd_silence
from pydub import effects as pd_effects

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".wma", ".aac"}

# ----------------- Utils -----------------
def find_audio_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS]

def ensure_mono16k(seg: AudioSegment) -> AudioSegment:
    if seg.channels != 1: seg = seg.set_channels(1)
    if seg.frame_rate != 16000: seg = seg.set_frame_rate(16000)
    if seg.sample_width != 2: seg = seg.set_sample_width(2)  # PCM16
    return seg

def pad_to_min_bounds(L: int, s: int, e: int, pad_ms: int, min_len_ms: int) -> Tuple[int, int]:
    s2 = max(0, s - pad_ms)
    e2 = min(L, e + pad_ms)
    need = min_len_ms - (e2 - s2)
    if need > 0:
        add_left = min(need // 2, s2)
        add_right = min(need - add_left, L - e2)
        s2 -= add_left
        e2 += add_right
    return s2, e2

def export_clip(src_path: Path, seg: AudioSegment, s: int, e: int, out_root: Path,
                pad_ms: int = 0, ensure_min_ms: int = 0) -> Tuple[Path, int, int]:
    L = len(seg)
    if ensure_min_ms or pad_ms:
        s, e = pad_to_min_bounds(L, s, e, pad_ms, ensure_min_ms or 0)
    rel = src_path.relative_to(ROOT_DIR).with_suffix("")  # mirror tree
    out_dir = out_root / rel
    out_dir.mkdir(parents=True, exist_ok=True)
    clip = seg[s:e]
    out_path = out_dir / f"{src_path.stem}_{s:06d}-{e:06d}.wav"
    clip.export(out_path, format="wav", parameters=["-acodec", "pcm_s16le"])
    return out_path, s, e

def write_csv_header(path: Path):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "source_path","start_ms","end_ms","slice_path","duration_ms",
            "mode","sil_min_len","sil_offset_db","sil_keep_ms","sil_join_ms",
            "max_ms","force_window_ms","pad_ms","ensure_min_ms",
            "merge_gap_ms","snap_ms","min_speech_ms"
        ])

def write_csv_rows(path: Path, rows: List[List]):
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(rows)

# ----------------- VADs -----------------
def detect_by_silence(
    seg: AudioSegment,
    min_silence_len: int = 220,
    silence_offset_db: float = 10.0,
    keep_ms: int = 200,
    join_ms: int = 220,
    min_clip_ms: int = 400,
    max_ms: int = 15000,
) -> List[Tuple[int,int]]:
    dbfs = seg.dBFS if seg.dBFS != float("-inf") else -50.0
    silence_thresh = dbfs - silence_offset_db
    regions = pd_silence.detect_nonsilent(
        seg,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh
    )
    merged = []
    for s, e in regions:
        if not merged:
            merged.append([s, e]); continue
        if s - merged[-1][1] <= join_ms:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    L = len(seg)
    out = []
    for s, e in merged:
        s2 = max(0, s - keep_ms)
        e2 = min(L, e + keep_ms)
        if e2 - s2 >= min_clip_ms:
            cur = s2
            while e2 - cur > max_ms:
                out.append((cur, cur + max_ms))
                cur += max_ms
            out.append((cur, e2))
    return out

def detect_by_webrtcvad(
    seg: AudioSegment,
    frame_ms: int = 30,
    aggressiveness: int = 2,
    keep_ms: int = 200,
    join_ms: int = 260,
    min_clip_ms: int = 400,
    max_ms: int = 15000,
) -> List[Tuple[int,int]]:
    try:
        import webrtcvad
    except Exception as e:
        raise RuntimeError("webrtcvad not installed. pip install webrtcvad") from e

    vad = webrtcvad.Vad(aggressiveness)
    sr = 16000
    seg = ensure_mono16k(seg)
    raw = seg.raw_data
    sw = seg.sample_width
    assert sw == 2 and seg.frame_rate == sr and seg.channels == 1

    frame_len = int(sr * (frame_ms / 1000.0))  # samples
    byte_len = frame_len * sw
    L = len(seg)

    speech_regions = []
    cur_start = None
    t = 0
    for i in range(0, len(raw) - byte_len + 1, byte_len):
        chunk = raw[i:i+byte_len]
        is_speech = vad.is_speech(chunk, sr)
        if is_speech and cur_start is None:
            cur_start = t
        elif (not is_speech) and cur_start is not None:
            speech_regions.append([cur_start, t])
            cur_start = None
        t += frame_ms
    if cur_start is not None:
        speech_regions.append([cur_start, t])

    merged = []
    for s, e in speech_regions:
        if not merged:
            merged.append([s, e]); continue
        if s - merged[-1][1] <= join_ms:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    out = []
    for s, e in merged:
        s2 = max(0, s - keep_ms)
        e2 = min(L, e + keep_ms)
        if e2 - s2 >= min_clip_ms:
            cur = s2
            while e2 - cur > max_ms:
                out.append((cur, cur + max_ms))
                cur += max_ms
            out.append((cur, e2))
    return out

# ----------------- Anti mid-word logic -----------------
def merge_close_gaps(ranges: List[Tuple[int,int]], merge_gap_ms: int) -> List[Tuple[int,int]]:
    if not ranges: return ranges
    merged = [list(ranges[0])]
    for s, e in ranges[1:]:
        if s - merged[-1][1] <= merge_gap_ms:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [tuple(x) for x in merged]

def drop_short_segments(ranges: List[Tuple[int,int]], min_speech_ms: int) -> List[Tuple[int,int]]:
    return [r for r in ranges if (r[1]-r[0]) >= min_speech_ms]

def _rms(seg: AudioSegment) -> float:
    # Use linear RMS (no dB) so minima comparisons are straightforward
    return seg.rms

def snap_boundaries_to_minima(seg: AudioSegment, ranges: List[Tuple[int,int]], snap_ms: int,
                              win_ms: int = 24, step_ms: int = 4) -> List[Tuple[int,int]]:
    """
    For each boundary between two consecutive ranges, search ±snap_ms around the cut
    and move boundary to the point with minimum RMS in a small window (win_ms).
    """
    if snap_ms <= 0 or not ranges: return ranges
    L = len(seg)
    snapped = [list(ranges[0])]

    def local_min(center_ms: int) -> int:
        start = max(0, center_ms - snap_ms)
        end = min(L, center_ms + snap_ms)
        best_t = center_ms
        best_val = None
        t = start
        half = win_ms // 2
        while t <= end:
            s = max(0, t - half); e = min(L, t + half)
            if e > s:
                val = _rms(seg[s:e])
                if (best_val is None) or (val < best_val):
                    best_val = val; best_t = t
            t += step_ms
        return best_t

    for i in range(1, len(ranges)):
        prev_s, prev_e = snapped[-1]
        cur_s, cur_e = ranges[i]
        # propose snapping at the existing boundary (prev_e == cur_s ideally)
        cut = int((prev_e + cur_s)//2) if prev_e != cur_s else prev_e
        new_cut = local_min(cut)
        # Adjust
        prev_e2 = max(prev_s, min(new_cut, L))
        cur_s2  = min(cur_e, max(new_cut, 0))
        # ensure no inversion
        if cur_s2 < prev_e2:
            mid = (prev_e2 + cur_s2)//2
            prev_e2 = mid
            cur_s2 = mid
        snapped[-1][1] = prev_e2
        snapped.append([cur_s2, cur_e])
    return [tuple(x) for x in snapped]

def force_window_split(ranges: List[Tuple[int,int]], window_ms: int) -> List[Tuple[int,int]]:
    if window_ms <= 0: return ranges
    out = []
    for s, e in ranges:
        cur = s
        while e - cur > window_ms:
            out.append((cur, cur + window_ms))
            cur += window_ms
        if e - cur > 0:
            out.append((cur, e))
    return out

# ----------------- Main -----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Short clip splitter with anti mid-word cuts (snapping & smoothing)")
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)

    # Pre-filters
    ap.add_argument("--pre_highpass", type=int, default=120)
    ap.add_argument("--pre_lowpass", type=int, default=8000)
    ap.add_argument("--pre_normalize", action="store_true")

    # VAD choice
    ap.add_argument("--vad", type=str, default="webrtc", choices=["silence","webrtc"])
    # pydub
    ap.add_argument("--silence_min_len", type=int, default=240)
    ap.add_argument("--silence_offset_db", type=float, default=10.0)
    ap.add_argument("--silence_keep_ms", type=int, default=200)
    ap.add_argument("--silence_join_ms", type=int, default=260)
    # webrtc
    ap.add_argument("--vad_aggressiveness", type=int, default=2, choices=[0,1,2,3])
    ap.add_argument("--vad_frame_ms", type=int, default=30, choices=[10,20,30])

    # Output constraints
    ap.add_argument("--min_clip_ms", type=int, default=400)
    ap.add_argument("--max_ms", type=int, default=15000)

    # Anti mid-word params
    ap.add_argument("--merge_gap_ms", type=int, default=240, help="Merge neighbor regions if gap <= this")
    ap.add_argument("--snap_ms", type=int, default=60, help="Search radius around boundary to snap to energy minimum (0=off)")
    ap.add_argument("--min_speech_ms", type=int, default=500, help="Drop segments shorter than this after smoothing")

    # Extra micro-chunking (typically OFF to avoid mid-word cuts)
    ap.add_argument("--force_window_ms", type=int, default=0)

    # Export padding
    ap.add_argument("--pad_ms", type=int, default=180)
    ap.add_argument("--ensure_min_ms", type=int, default=0)

    args = ap.parse_args()

    global ROOT_DIR
    ROOT_DIR = Path(args.root)
    OUT_ROOT = Path(args.out)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_ROOT / "manifest_no_midword.csv"
    write_csv_header(csv_path)

    files = find_audio_files(ROOT_DIR)
    if not files:
        print("No audio files found.")
        raise SystemExit

    total = 0
    for p in tqdm(files, desc="Processing", unit="file"):
        try:
            seg0 = AudioSegment.from_file(p)
            seg = ensure_mono16k(seg0)

            # Pre-filter
            if args.pre_highpass and args.pre_highpass > 0:
                seg = seg.high_pass_filter(args.pre_highpass)
            if args.pre_lowpass and args.pre_lowpass > 0:
                seg = seg.low_pass_filter(args.pre_lowpass)
            if args.pre_normalize:
                seg = pd_effects.normalize(seg)

            # VAD
            if args.vad == "webrtc":
                ranges = detect_by_webrtcvad(
                    seg,
                    frame_ms=args.vad_frame_ms,
                    aggressiveness=args.vad_aggressiveness,
                    keep_ms=args.silence_keep_ms,
                    join_ms=args.silence_join_ms,
                    min_clip_ms=args.min_clip_ms,
                    max_ms=args.max_ms,
                )
                mode = "webrtc"
            else:
                ranges = detect_by_silence(
                    seg,
                    min_silence_len=args.silence_min_len,
                    silence_offset_db=args.silence_offset_db,
                    keep_ms=args.silence_keep_ms,
                    join_ms=args.silence_join_ms,
                    min_clip_ms=args.min_clip_ms,
                    max_ms=args.max_ms
                )
                mode = "silence"

            if not ranges:
                continue

            # Anti mid-word smoothing
            ranges = merge_close_gaps(ranges, args.merge_gap_ms)
            ranges = snap_boundaries_to_minima(seg, ranges, args.snap_ms, win_ms=24, step_ms=4)
            ranges = drop_short_segments(ranges, args.min_speech_ms)

            # Optional fixed-window split (usually off to avoid mid-word cuts)
            ranges = force_window_split(ranges, args.force_window_ms)

            if not ranges:
                continue

            rows = []
            for s, e in ranges:
                out_wav, s2, e2 = export_clip(
                    p, seg, s, e, OUT_ROOT,
                    pad_ms=args.pad_ms, ensure_min_ms=args.ensure_min_ms
                )
                dur = e2 - s2
                rows.append([
                    str(p), s2, e2, str(out_wav), dur,
                    mode, args.silence_min_len, args.silence_offset_db, args.silence_keep_ms,
                    args.silence_join_ms, args.max_ms, args.force_window_ms, args.pad_ms, args.ensure_min_ms,
                    args.merge_gap_ms, args.snap_ms, args.min_speech_ms
                ])
                total += 1

            write_csv_rows(csv_path, rows)

        except Exception as ex:
            print(f"[WARN] {p}: {ex}")

    print(f"DONE. total clips: {total}")
    print(f"- CSV:   {csv_path.resolve()}")
    print(f"- OUT:   {OUT_ROOT.resolve()}")
