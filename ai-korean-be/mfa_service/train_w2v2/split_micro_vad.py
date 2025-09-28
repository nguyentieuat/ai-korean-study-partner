#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
split_micro_vad.py
Cut audio into very short clips using either pydub "silence" VAD or WebRTC VAD,
with optional pre-filtering and an extra fixed-window splitter for micro-chunks.

Dependencies:
  - pydub (requires ffmpeg in PATH)
  - tqdm
  - (optional) webrtcvad

Usage (example for "as short as possible"):
  python split_micro_vad.py \
    --root "E:/USE/My_project/source audio/audio train w2v2_16k" \
    --out  "E:/USE/My_project/source audio/output_micro" \
    --vad webrtc --vad_aggressiveness 3 --vad_frame_ms 20 \
    --silence_keep_ms 80 --silence_join_ms 80 \
    --min_clip_ms 150 --max_ms 2000 \
    --force_window_ms 500 \
    --pad_ms 120 \
    --pre_highpass 120 --pre_lowpass 8000 --pre_normalize
"""

import os, csv, json, argparse, re, unicodedata as ud
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm
from pydub import AudioSegment
from pydub import silence as pd_silence
from pydub import effects as pd_effects

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".wma", ".aac"}

# ----------------- Utils -----------------
def find_audio_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS]

def ensure_mono16k(seg: AudioSegment) -> AudioSegment:
    """Force PCM mono 16kHz 16-bit for stable processing."""
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
        ])

def write_csv_rows(path: Path, rows: List[List]):
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(rows)

# ----------------- VADs -----------------
def detect_by_silence(
    seg: AudioSegment,
    min_silence_len: int = 150,
    silence_offset_db: float = 10.0,
    keep_ms: int = 80,
    join_ms: int = 80,
    min_clip_ms: int = 150,
    max_ms: int = 2000,
) -> List[Tuple[int,int]]:
    """
    pydub energy-based VAD. Returns list (start_ms, end_ms).
    """
    dbfs = seg.dBFS if seg.dBFS != float("-inf") else -50.0
    silence_thresh = dbfs - silence_offset_db
    regions = pd_silence.detect_nonsilent(
        seg,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh
    )

    # Merge close regions
    merged = []
    for s, e in regions:
        if not merged:
            merged.append([s, e]); continue
        if s - merged[-1][1] <= join_ms:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    # Expand edges & filter by length; also chunk by max_ms
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
    frame_ms: int = 20,
    aggressiveness: int = 3,
    keep_ms: int = 80,
    join_ms: int = 80,
    min_clip_ms: int = 150,
    max_ms: int = 2000,
) -> List[Tuple[int,int]]:
    """
    WebRTC VAD (robust in noise). Returns list (start_ms, end_ms).
    """
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

    # merge close
    merged = []
    for s, e in speech_regions:
        if not merged:
            merged.append([s, e]); continue
        if s - merged[-1][1] <= join_ms:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    # expand + chunk + filter
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

def force_window_split(ranges: List[Tuple[int,int]], window_ms: int) -> List[Tuple[int,int]]:
    """
    Further split each (s,e) into fixed windows of window_ms (last chunk shorter allowed).
    """
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
    ap = argparse.ArgumentParser(description="Micro VAD splitter (pydub silence / WebRTC) + fixed-window micro-chunks")
    ap.add_argument("--root", type=str, required=True, help="Root folder of audios")
    ap.add_argument("--out", type=str, required=True, help="Output folder")
    # Pre-filters
    ap.add_argument("--pre_highpass", type=int, default=120, help="HPF Hz (0 = off)")
    ap.add_argument("--pre_lowpass", type=int, default=8000, help="LPF Hz (0 = off)")
    ap.add_argument("--pre_normalize", action="store_true", help="Normalize RMS before VAD")

    # VAD choice
    ap.add_argument("--vad", type=str, default="silence", choices=["silence","webrtc"],
                    help="silence = pydub energy; webrtc = WebRTC VAD")
    # pydub silence params
    ap.add_argument("--silence_min_len", type=int, default=150)
    ap.add_argument("--silence_offset_db", type=float, default=8.0)
    ap.add_argument("--silence_keep_ms", type=int, default=80)
    ap.add_argument("--silence_join_ms", type=int, default=80)

    # webrtc params
    ap.add_argument("--vad_aggressiveness", type=int, default=3, choices=[0,1,2,3])
    ap.add_argument("--vad_frame_ms", type=int, default=20, choices=[10,20,30])

    # Output constraints
    ap.add_argument("--min_clip_ms", type=int, default=150)
    ap.add_argument("--max_ms", type=int, default=2000)

    # Extra micro-chunking
    ap.add_argument("--force_window_ms", type=int, default=500, help="Further split each region into fixed windows (0=off)")

    # Export padding
    ap.add_argument("--pad_ms", type=int, default=120, help="Pad when exporting (ms)")
    ap.add_argument("--ensure_min_ms", type=int, default=0, help="Ensure clip >= value (ms)")

    args = ap.parse_args()

    global ROOT_DIR
    ROOT_DIR = Path(args.root)
    OUT_ROOT = Path(args.out)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_ROOT / "manifest_micro.csv"
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

            # Extra fixed-window micro-chunking
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
                    args.silence_join_ms, args.max_ms, args.force_window_ms, args.pad_ms, args.ensure_min_ms
                ])
                total += 1

            write_csv_rows(csv_path, rows)

        except Exception as ex:
            print(f"[WARN] {p}: {ex}")

    print(f"DONE. total clips: {total}")
    print(f"- CSV:   {csv_path.resolve()}")
    print(f"- OUT:   {OUT_ROOT.resolve()}")
