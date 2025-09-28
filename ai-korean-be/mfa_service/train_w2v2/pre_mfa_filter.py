#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pre_mfa_filter.py
Filter out silent/low-speech clips and trim long silences BEFORE MFA.

Features
- Detect speech ratio with WebRTC VAD (fallback: energy threshold)
- Drop clips with too little speech or too short/too long
- Trim long leading/trailing silences to the first/last speech (±pad)
- (Optional) Remove long internal silences (concatenate surrounding speech)
- Write a CSV manifest with decisions and durations

Requirements
  pip install pydub webrtcvad tqdm

Usage (typical):
  python pre_mfa_filter.py \
    --root "E:/USE/My_project/source audio/output_vad" \
    --out  "E:/USE/My_project/source audio/pre_mfa_clean" \
    --min_speech_ratio 0.25 \
    --max_lead_ms 1000 --max_tail_ms 1000 \
    --trim_pad_ms 150 --min_silence_len 300 --silence_offset_db 12

If you want to also delete very long internal silences (e.g., > 1500 ms):
  python pre_mfa_filter.py ... --internal_remove_ms 1500
"""

import csv, argparse, math
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm
from pydub import AudioSegment, silence as pd_silence, effects as pd_effects

# Optional: WebRTC VAD
try:
    import webrtcvad
except Exception:
    webrtcvad = None

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".wma", ".aac"}

def find_audio_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS]

def ensure_mono16k(seg: AudioSegment) -> AudioSegment:
    if seg.channels != 1: seg = seg.set_channels(1)
    if seg.frame_rate != 16000: seg = seg.set_frame_rate(16000)
    if seg.sample_width != 2: seg = seg.set_sample_width(2)
    return seg

def compute_vad_speech_ratio(seg: AudioSegment, frame_ms: int = 30, aggressiveness: int = 2) -> Tuple[float, List[Tuple[int,int]]]:
    """
    Returns (speech_ratio, speech_regions_ms) using WebRTC VAD if available,
    else falls back to energy-based nonsilent detection.
    """
    if webrtcvad is None:
        # energy fallback
        dbfs = seg.dBFS if seg.dBFS != float("-inf") else -50.0
        silence_thresh = dbfs - 12.0
        regions = pd_silence.detect_nonsilent(seg, min_silence_len=200, silence_thresh=silence_thresh)
        L = len(seg)
        speech_ms = sum((e - s) for s, e in regions)
        ratio = max(0.0, min(1.0, speech_ms / max(1, L)))
        return ratio, [(s, e) for s, e in regions]
    # WebRTC path
    vad = webrtcvad.Vad(aggressiveness)
    seg = ensure_mono16k(seg)
    sr = seg.frame_rate
    sw = seg.sample_width
    raw = seg.raw_data
    frame_len = int(sr * (frame_ms / 1000.0))  # samples
    byte_len = frame_len * sw
    speech_regions = []
    cur_start = None
    t = 0
    for i in range(0, len(raw) - byte_len + 1, byte_len):
        chunk = raw[i:i+byte_len]
        is_speech = vad.is_speech(chunk, sr)
        if is_speech and cur_start is None:
            cur_start = t
        elif (not is_speech) and cur_start is not None:
            speech_regions.append((cur_start, t))
            cur_start = None
        t += frame_ms
    if cur_start is not None:
        speech_regions.append((cur_start, t))
    L = len(seg)
    speech_ms = sum((e - s) for s, e in speech_regions)
    ratio = max(0.0, min(1.0, speech_ms / max(1, L)))
    return ratio, speech_regions

def detect_nonsilent_regions(seg: AudioSegment, min_silence_len: int, silence_offset_db: float) -> List[Tuple[int,int]]:
    dbfs = seg.dBFS if seg.dBFS != float("-inf") else -50.0
    silence_thresh = dbfs - silence_offset_db
    regions = pd_silence.detect_nonsilent(seg, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    return [(max(0,s), min(len(seg),e)) for s,e in regions]

def trim_edges(seg: AudioSegment, regions: List[Tuple[int,int]], pad_ms: int) -> Tuple[int,int]:
    """Return start,end after trimming to first/last nonsilent region with pad."""
    if not regions:
        return 0, len(seg)
    s = max(0, regions[0][0] - pad_ms)
    e = min(len(seg), regions[-1][1] + pad_ms)
    return s, e

def remove_internal_silences(seg: AudioSegment, regions: List[Tuple[int,int]], internal_remove_ms: int) -> AudioSegment:
    """Concatenate speech regions, but keep gaps < internal_remove_ms (to preserve micro-pauses)."""
    if internal_remove_ms <= 0 or not regions:
        return seg
    pieces = []
    prev_end = 0
    for (s, e) in regions:
        if pieces and (s - prev_end) < internal_remove_ms:
            # keep the small gap
            pieces.append(seg[prev_end:s])
        # append speech
        pieces.append(seg[s:e])
        prev_end = e
    combined = AudioSegment.empty()
    for part in pieces:
        combined += part
    return combined

def main():
    ap = argparse.ArgumentParser(description="Filter/trim clips before MFA")
    ap.add_argument("--root", required=True, help="Input folder with audio clips")
    ap.add_argument("--out", required=True, help="Output folder for cleaned clips")
    ap.add_argument("--min_dur_ms", type=int, default=300)
    ap.add_argument("--max_dur_ms", type=int, default=0, help="0 = no upper limit")
    ap.add_argument("--min_speech_ratio", type=float, default=0.25, help="Drop if speech frames / total < this")
    ap.add_argument("--frame_ms", type=int, default=30, help="VAD frame size (10/20/30)")
    ap.add_argument("--vad_aggr", type=int, default=2, choices=[0,1,2,3])
    ap.add_argument("--min_silence_len", type=int, default=300, help="For energy-based nonsilent detection")
    ap.add_argument("--silence_offset_db", type=float, default=12.0, help="Thresh = dBFS - offset")
    ap.add_argument("--max_lead_ms", type=int, default=1200, help="If leading silence > this, we will trim to first speech (±pad)")
    ap.add_argument("--max_tail_ms", type=int, default=1200, help="If trailing silence > this, we will trim to last speech (±pad)")
    ap.add_argument("--trim_pad_ms", type=int, default=150)
    ap.add_argument("--internal_remove_ms", type=int, default=0, help="Remove internal silence >= this (0 = don't remove)")
    ap.add_argument("--normalize", action="store_true", help="Peak normalize after trimming")
    ap.add_argument("--move_reject", action="store_true", help="Move rejected files to out/rejected instead of ignoring")
    args = ap.parse_args()

    ROOT = Path(args.root)
    OUT  = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    REJ = OUT / "rejected"
    if args.move_reject:
        REJ.mkdir(parents=True, exist_ok=True)

    csv_path = OUT / "pre_mfa_clean_manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["orig_path","out_path","action","reason",
                    "orig_ms","new_ms","speech_ratio","lead_ms","tail_ms","max_internal_sil_ms"])

    files = find_audio_files(ROOT)
    total = 0
    kept = 0
    dropped = 0
    for p in tqdm(files, desc="Pre-MFA filtering", unit="file"):
        try:
            seg0 = AudioSegment.from_file(p)
            seg = ensure_mono16k(seg0)
            L = len(seg)
            if L < args.min_dur_ms or (args.max_dur_ms and L > args.max_dur_ms):
                action = "drop"
                reason = "too_short" if L < args.min_dur_ms else "too_long"
                out_path = ""
                if args.move_reject:
                    out_path = str((REJ / p.name).with_suffix(".wav"))
                    seg.export(out_path, format="wav", parameters=["-acodec","pcm_s16le"])
                with csv_path.open("a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([str(p), out_path, action, reason, L, 0, 0, "", "", ""])
                dropped += 1; total += 1; continue

            # VAD/energy analysis
            speech_ratio, vad_regions = compute_vad_speech_ratio(seg, frame_ms=args.frame_ms, aggressiveness=args.vad_aggr)
            # energy nonsilent regions for precise edges
            ns_regions = detect_nonsilent_regions(seg, args.min_silence_len, args.silence_offset_db)
            lead_ms = ns_regions[0][0] if ns_regions else L
            tail_ms = (L - ns_regions[-1][1]) if ns_regions else L
            # max internal silence (via complement of ns_regions)
            max_internal = 0
            if ns_regions:
                prev_e = ns_regions[0][1]
                for s, e in ns_regions[1:]:
                    gap = max(0, s - prev_e)
                    if gap > max_internal:
                        max_internal = gap
                    prev_e = e

            # Decide drop if no speech
            if speech_ratio < args.min_speech_ratio or not ns_regions:
                action = "drop"
                reason = "low_speech_ratio" if speech_ratio < args.min_speech_ratio else "no_nonsilent"
                out_path = ""
                if args.move_reject:
                    out_path = str((REJ / p.name).with_suffix(".wav"))
                    seg.export(out_path, format="wav", parameters=["-acodec","pcm_s16le"])
                with csv_path.open("a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([str(p), out_path, action, reason, L, 0, round(speech_ratio,4), lead_ms, tail_ms, max_internal])
                dropped += 1; total += 1; continue

            # Trim edges if long silence
            s_trim, e_trim = 0, L
            need_trim_edges = (lead_ms > args.max_lead_ms) or (tail_ms > args.max_tail_ms)
            if need_trim_edges:
                s_trim, e_trim = trim_edges(seg, ns_regions, pad_ms=args.trim_pad_ms)
            trimmed = seg[s_trim:e_trim]

            # Remove internal long silences if requested
            if args.internal_remove_ms and max_internal >= args.internal_remove_ms:
                trimmed = remove_internal_silences(trimmed, [(max(0,s - s_trim), max(0,e - s_trim)) for (s,e) in ns_regions
                                                             if e > s_trim and s < e_trim],
                                                   internal_remove_ms=args.internal_remove_ms)

            if args.normalize:
                trimmed = pd_effects.normalize(trimmed)

            # Export
            rel = p.relative_to(ROOT)
            out_dir = (OUT / rel.parent)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_wav = (out_dir / rel.name).with_suffix(".wav")
            trimmed.export(out_wav, format="wav", parameters=["-acodec","pcm_s16le"])

            with csv_path.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([str(p), str(out_wav), "keep", "trimmed" if need_trim_edges else "as_is",
                                        L, len(trimmed), round(speech_ratio,4), lead_ms, tail_ms, max_internal])
            kept += 1; total += 1

        except Exception as ex:
            with csv_path.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([str(p), "", "error", str(ex), "", "", "", "", "", ""])

    print(f"Done. total={total}, kept={kept}, dropped={dropped}")
    print("CSV:", csv_path.resolve())
    print("OUT:", OUT.resolve())

if __name__ == "__main__":
    main()
