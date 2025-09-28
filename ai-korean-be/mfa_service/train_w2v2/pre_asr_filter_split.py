#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pre_asr_filter_split.py  (subfolder-preserving)
Filter out silent/low-speech clips and optionally SPLIT clips with long internal silences
BEFORE running ASR / ko_phonology / MFA.

What's new vs previous:
- **Preserve subfolder structure** for all outputs (and rejected files if desired).
- Accept either --root (folder) or --from_manifest (CSV of files).
- Auto-detect a base root (common parent) when only a manifest is provided, or allow --root_base.

Key features
- Speech ratio via WebRTC VAD (fallback: energy-based nonsilent detection)
- Drop clips with too little speech or out-of-range duration
- Trim long leading/trailing silences to first/last speech (±pad)
- Either REMOVE very long internal silences (concatenate) OR SPLIT into multiple clips at long gaps
- Export processed WAVs & CSV manifests: a summary + a per-piece mapping

Requirements
  pip install pydub webrtcvad tqdm

Examples
  python pre_asr_filter_split.py \
    --root "E:/USE/My_project/source audio/output_vad" \
    --out  "E:/USE/My_project/pre_asr_clean" \
    --min_speech_ratio 0.25 \
    --max_lead_ms 1000 --max_tail_ms 1000 --trim_pad_ms 150 \
    --min_silence_len 300 --silence_offset_db 12 \
    --internal_split_ms 1500 --internal_join_ms 220 \
    --min_dur_ms 350 --max_dur_ms 12000 \
    --normalize

  python pre_asr_filter_split.py \
    --from_manifest "E:/.../manifest_no_midword.csv" \
    --out "E:/.../pre_asr_clean" \
    --internal_split_ms 1500 --internal_join_ms 220 --min_dur_ms 350 --max_dur_ms 12000
"""

import os, csv, argparse, shutil
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

# ---------------- IO helpers ----------------
def list_from_folder(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS]

def list_from_manifest(csv_path: Path) -> List[Path]:
    paths = []
    with csv_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            return paths
        col = None
        for cand in ["slice_path","path","audio","wav","file","filepath"]:
            if cand in r.fieldnames:
                col = cand; break
        if col is None:
            raise RuntimeError("Manifest must have one of columns: slice_path/path/audio/wav/file/filepath")
        for row in r:
            p = Path(row[col])
            if p.exists():
                paths.append(p)
    return paths

def ensure_mono16k(seg: AudioSegment) -> AudioSegment:
    if seg.channels != 1: seg = seg.set_channels(1)
    if seg.frame_rate != 16000: seg = seg.set_frame_rate(16000)
    if seg.sample_width != 2: seg = seg.set_sample_width(2)  # 16-bit
    return seg

# --------------- detection ---------------
def energy_nonsilent(seg: AudioSegment, min_silence_len: int, silence_offset_db: float) -> List[Tuple[int,int]]:
    dbfs = seg.dBFS if seg.dBFS != float("-inf") else -50.0
    silence_thresh = dbfs - silence_offset_db
    regions = pd_silence.detect_nonsilent(seg, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    return [(max(0,s), min(len(seg),e)) for s,e in regions]

def speech_ratio_vad(seg: AudioSegment, frame_ms: int = 30, aggressiveness: int = 2) -> Tuple[float, List[Tuple[int,int]]]:
    """Return (speech_ratio, speech_regions). If webrtcvad unavailable, fall back to energy-based."""
    if webrtcvad is None:
        regions = energy_nonsilent(seg, min_silence_len=200, silence_offset_db=12.0)
        speech_ms = sum(e-s for s,e in regions)
        return (speech_ms / max(1,len(seg))), regions
    vad = webrtcvad.Vad(aggressiveness)
    seg = ensure_mono16k(seg)
    sr = seg.frame_rate
    sw = seg.sample_width
    raw = seg.raw_data
    frame_len_samples = int(sr * (frame_ms/1000.0))
    frame_len_bytes = frame_len_samples * sw
    regions = []
    cur_start = None
    t = 0
    for i in range(0, len(raw) - frame_len_bytes + 1, frame_len_bytes):
        chunk = raw[i:i+frame_len_bytes]
        is_speech = vad.is_speech(chunk, sr)
        if is_speech and cur_start is None:
            cur_start = t
        elif (not is_speech) and cur_start is not None:
            regions.append((cur_start, t))
            cur_start = None
        t += frame_ms
    if cur_start is not None:
        regions.append((cur_start, t))
    speech_ms = sum(e-s for s,e in regions)
    return (speech_ms / max(1,len(seg))), regions

# --------------- processing ---------------
def trim_to_speech_edges(seg: AudioSegment, ns_regions: List[Tuple[int,int]], pad_ms: int) -> Tuple[int,int]:
    if not ns_regions:
        return 0, len(seg)
    s = max(0, ns_regions[0][0] - pad_ms)
    e = min(len(seg), ns_regions[-1][1] + pad_ms)
    return s, e

def group_regions_by_gap(ns: List[Tuple[int,int]], join_gap_ms: int, split_gap_ms: int) -> List[Tuple[int,int]]:
    """
    Merge regions closer than join_gap_ms; start a new chunk when a gap >= split_gap_ms.
    Return merged chunks (start,end) in ms.
    """
    out = []
    if not ns: return out
    merged = [list(ns[0])]
    for s,e in ns[1:]:
        if s - merged[-1][1] <= join_gap_ms:
            merged[-1][1] = e
        else:
            merged.append([s,e])
    cur_s, cur_e = merged[0]
    for s,e in merged[1:]:
        gap = s - cur_e
        if gap >= split_gap_ms:
            out.append((cur_s, cur_e))
            cur_s, cur_e = s, e
        else:
            cur_e = e
    out.append((cur_s, cur_e))
    return out

def common_parent(paths: List[Path]) -> Path:
    if not paths:
        return Path(".").resolve()
    try:
        cp = os.path.commonpath([str(p.resolve()) for p in paths])
        return Path(cp)
    except Exception:
        return paths[0].resolve().parent

def rel_under(base: Path, p: Path) -> Path:
    try:
        return p.resolve().relative_to(base.resolve())
    except Exception:
        return Path(p.name)

def save_wav(seg: AudioSegment, src_rel: Path, base_out: Path, s: int, e: int, normalize: bool = False) -> Path:
    cut = seg[s:e]
    if normalize:
        cut = pd_effects.normalize(cut)
    # preserve tree under base_out / src_rel.parent
    out_dir = (base_out / src_rel.parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{src_rel.stem}_{s:06d}-{e:06d}.wav"
    out_path = out_dir / name
    cut.export(out_path, format="wav", parameters=["-acodec", "pcm_s16le"])
    return out_path

def copy_rejected(src_abs: Path, src_rel: Path, out_root: Path, move_reject: bool) -> Path:
    rej_dir = out_root / "rejected" / src_rel.parent
    rej_dir.mkdir(parents=True, exist_ok=True)
    dst = rej_dir / src_rel.name
    if move_reject:
        try:
            shutil.move(str(src_abs), str(dst))
        except Exception:
            shutil.copy2(str(src_abs), str(dst))
    else:
        shutil.copy2(str(src_abs), str(dst))
    return dst

def main():
    ap = argparse.ArgumentParser(description="Filter & re-split clips before ASR (preserve subfolders)")
    ap.add_argument("--root", type=str, default=None, help="Folder to scan for audio (recursive)")
    ap.add_argument("--from_manifest", type=str, default=None, help="CSV manifest with a path column (slice_path/path/audio/...)")
    ap.add_argument("--root_base", type=str, default=None, help="Optional base root for relative paths (when using manifest)")
    ap.add_argument("--out", required=True, help="Output folder for processed clips")
    # Speech & energy params
    ap.add_argument("--min_speech_ratio", type=float, default=0.25, help="Drop if speech_ratio < this")
    ap.add_argument("--vad_frame_ms", type=int, default=30, choices=[10,20,30])
    ap.add_argument("--vad_aggr", type=int, default=2, choices=[0,1,2,3])
    ap.add_argument("--min_silence_len", type=int, default=300, help="Energy-based detection param")
    ap.add_argument("--silence_offset_db", type=float, default=12.0, help="Thresh = dBFS - offset")
    # Edge trimming
    ap.add_argument("--max_lead_ms", type=int, default=1200)
    ap.add_argument("--max_tail_ms", type=int, default=1200)
    ap.add_argument("--trim_pad_ms", type=int, default=150)
    # Internal gaps
    ap.add_argument("--internal_split_ms", type=int, default=0, help="Split at internal gaps >= this (0 = don't split)")
    ap.add_argument("--internal_join_ms", type=int, default=220, help="Join gaps < this before splitting/removing")
    ap.add_argument("--internal_remove_ms", type=int, default=0, help="Remove internal silence >= this (0 = don't remove)")
    # Durations
    ap.add_argument("--min_dur_ms", type=int, default=300)
    ap.add_argument("--max_dur_ms", type=int, default=0, help="0 = no upper limit")
    # Output processing
    ap.add_argument("--normalize", action="store_true", help="Normalize each output clip")
    ap.add_argument("--move_reject", action="store_true", help="Also copy/move rejected files under out/rejected/ preserving subfolders")
    args = ap.parse_args()

    if not args.root and not args.from_manifest:
        raise SystemExit("Provide --root or --from_manifest")
    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)

    # Prepare list of files
    files: List[Path] = []
    if args.from_manifest:
        files = list_from_manifest(Path(args.from_manifest))
    if args.root:
        files += list_from_folder(Path(args.root))
    files = sorted(set(files))

    # Determine base for relative paths
    if args.root:
        base_root = Path(args.root).resolve()
    elif args.root_base:
        base_root = Path(args.root_base).resolve()
    else:
        base_root = common_parent(files)

    # CSV manifests
    summary_csv = out_root / "pre_asr_filter_split_manifest.csv"
    pieces_csv  = out_root / "pre_asr_filter_split_pieces.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["orig_path","action","reason","orig_ms","out_count","total_out_ms","speech_ratio","max_internal_gap_ms"])
    with pieces_csv.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["orig_path","out_path","start_ms","end_ms","duration_ms","reason"])

    kept_total = 0
    total = 0
    for p in tqdm(files, desc="Filtering/splitting", unit="file"):
        src_abs = p.resolve()
        src_rel = rel_under(base_root, src_abs)

        try:
            seg0 = AudioSegment.from_file(src_abs)
            seg = ensure_mono16k(seg0)
            L = len(seg)

            # VAD speech ratio
            speech_ratio, vad_regions = speech_ratio_vad(seg, frame_ms=args.vad_frame_ms, aggressiveness=args.vad_aggr)

            # Energy nonsilent for precise edge/gap analysis
            ns_regions = energy_nonsilent(seg, args.min_silence_len, args.silence_offset_db)

            # Decide reject for too little speech
            if speech_ratio < args.min_speech_ratio or not ns_regions:
                if args.move_reject:
                    copy_rejected(src_abs, src_rel, out_root, move_reject=True)
                with summary_csv.open("a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([str(src_rel), "drop", "low_speech" if speech_ratio < args.min_speech_ratio else "no_nonsilent", L, 0, 0, round(speech_ratio,4), ""])
                total += 1
                continue

            # Edge trim if leading/trailing silence is long
            lead_ms = ns_regions[0][0] if ns_regions else L
            tail_ms = (L - ns_regions[-1][1]) if ns_regions else L
            s_trim, e_trim = 0, L
            need_edge_trim = (lead_ms > args.max_lead_ms) or (tail_ms > args.max_tail_ms)
            if need_edge_trim:
                s_trim, e_trim = trim_to_speech_edges(seg, ns_regions, pad_ms=args.trim_pad_ms)
            seg2 = seg[s_trim:e_trim]
            ns2 = [(max(0,s - s_trim), max(0,e - s_trim)) for s,e in ns_regions if e > s_trim and s < e_trim]

            # Internal handling
            out_ranges: List[Tuple[int,int]] = []
            reason = "as_is"
            max_gap = 0
            if ns2:
                prev_e = ns2[0][1]
                for s,e in ns2[1:]:
                    gap = max(0, s - prev_e)
                    if gap > max_gap: max_gap = gap
                    prev_e = e

                if args.internal_split_ms and max_gap >= args.internal_split_ms:
                    out_ranges = group_regions_by_gap(ns2, join_gap_ms=args.internal_join_ms, split_gap_ms=args.internal_split_ms)
                    reason = f"split@{args.internal_split_ms}"
                elif args.internal_remove_ms and max_gap >= args.internal_remove_ms:
                    chunks = group_regions_by_gap(ns2, join_gap_ms=args.internal_join_ms, split_gap_ms=10**9)
                    from pydub import AudioSegment as _AS
                    seg_combined = _AS.empty()
                    for s,e in chunks:
                        seg_combined += seg2[s:e]
                    seg2 = seg_combined
                    out_ranges = [(0, len(seg2))]
                    reason = f"remove@{args.internal_remove_ms}"
                else:
                    out_ranges = [(0, len(seg2))]
                    reason = "trimmed" if need_edge_trim else "as_is"

            # Enforce duration limits and export
            piece_paths = []
            total_out_ms = 0
            for (s,e) in out_ranges:
                dur = e - s
                if dur < args.min_dur_ms:
                    continue
                if args.max_dur_ms and dur > args.max_dur_ms:
                    cur = s
                    while e - cur > args.max_dur_ms:
                        a, b = cur, cur + args.max_dur_ms
                        path = save_wav(seg2, src_rel, out_root, a, b, normalize=args.normalize)
                        with pieces_csv.open("a", newline="", encoding="utf-8") as f:
                            csv.writer(f).writerow([str(src_rel), str(path.relative_to(out_root)), a, b, (b-a), reason])
                        piece_paths.append(path)
                        total_out_ms += (b - a)
                        cur = b
                    if e - cur >= args.min_dur_ms:
                        path = save_wav(seg2, src_rel, out_root, cur, e, normalize=args.normalize)
                        with pieces_csv.open("a", newline="", encoding="utf-8") as f:
                            csv.writer(f).writerow([str(src_rel), str(path.relative_to(out_root)), cur, e, (e-cur), reason])
                        piece_paths.append(path)
                        total_out_ms += (e - cur)
                else:
                    path = save_wav(seg2, src_rel, out_root, s, e, normalize=args.normalize)
                    with pieces_csv.open("a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([str(src_rel), str(path.relative_to(out_root)), s, e, dur, reason])
                    piece_paths.append(path)
                    total_out_ms += dur

            action = "keep" if piece_paths else "drop"
            with summary_csv.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([str(src_rel), action, reason, L, len(piece_paths), total_out_ms, round(speech_ratio,4), max_gap])

            if piece_paths:
                kept_total += 1
            total += 1

        except Exception as ex:
            with summary_csv.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([str(src_rel), "error", str(ex), "", "", "", "", ""])
            total += 1

    print(f"Done. files={total}, kept={kept_total}, out_dir={out_root.resolve()}")
    print("Summary :", summary_csv.resolve())
    print("Pieces  :", pieces_csv.resolve())

if __name__ == "__main__":
    main()
