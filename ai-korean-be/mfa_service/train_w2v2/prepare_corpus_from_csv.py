#!/usr/bin/env python
# prepare_corpus_from_csv.py  (filtered + number/unit verbalizer)
# Build an MFA corpus folder (wav + lab) từ CSV như slices_asr.csv,
# CHỈ bao gồm những dòng có text đạt yêu cầu.
#
# Usage (ví dụ):
#   python prepare_corpus_from_csv.py \
#       --csv /kaggle/input/your/slices_asr.csv \
#       --out_dir /kaggle/tmp/corpus \
#       --text_col text --audio_col audio_path \
#       --symlink 1 \
#       --drop_empty 1 --keep_punct_only 0 --drop_nonhangul 0 --min_chars 1 \
#       --verbalize_numbers 1 --native_counters 1 --no_celsius_prefix 0

import csv, argparse, shutil, os, re
from pathlib import Path

from ko_num_unit_verbalizer import verbalize_ko_numbers_units

_HANGUL_RE = re.compile(r"[\uAC00-\uD7A3]")
_PUNCT_ONLY_RE = re.compile(r"^[\s\.,?!;:\"'\(\)\[\]\{\}~\-–—…·]+$")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def is_meaningful_text(text: str, min_chars: int, drop_nonhangul: bool, keep_punct_only: bool) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if len(t) < min_chars:
        return False
    if not keep_punct_only and _PUNCT_ONLY_RE.match(t):
        return False
    if drop_nonhangul and not _HANGUL_RE.search(t):
        return False
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to slices_asr.csv")
    ap.add_argument("--out_dir", default="/kaggle/tmp/corpus")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--audio_col", default="audio_path")
    ap.add_argument("--symlink", type=int, default=1, help="1=symlink wavs, 0=copy")
    # filters
    ap.add_argument("--drop_empty", type=int, default=1)
    ap.add_argument("--keep_punct_only", type=int, default=0)
    ap.add_argument("--drop_nonhangul", type=int, default=0)
    ap.add_argument("--min_chars", type=int, default=1)
    # number/unit verbalizer
    ap.add_argument("--verbalize_numbers", type=int, default=1, help="1=đọc số/đơn vị ra chữ Hàn")
    ap.add_argument("--native_counters", type=int, default=1, help="1=dùng Thuần Hàn cho counter (시/시간/달/살/번/개/명/...)")
    ap.add_argument("--no_celsius_prefix", type=int, default=0, help="1=bỏ tiền tố '섭씨' trước °C")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    n_ok = 0
    n_skip_missing = 0
    n_skip_filtered = 0
    n_verbalized_changed = 0

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            audio = row.get(args.audio_col) or row.get("audio") or row.get("path") or ""
            text  = (row.get(args.text_col) or "").strip()

            # Bỏ row thiếu audio hoặc thiếu text (tuỳ cờ)
            if not audio:
                n_skip_missing += 1
                continue
            if int(args.drop_empty) and not text:
                n_skip_filtered += 1
                continue

            # >>> ĐỌC SỐ/ĐƠN VỊ TRƯỚC RỒI MỚI LỌC <<<
            if int(args.verbalize_numbers):
                t2 = verbalize_ko_numbers_units(
                    text,
                    use_native_counters=bool(args.native_counters),
                    include_celsius_prefix=not bool(args.no_celsius_prefix),
                )
                if t2 != text:
                    n_verbalized_changed += 1
                text = t2

            # Áp các filter sau verbalize (để không bỏ sót câu chỉ-là-số)
            if not is_meaningful_text(text, args.min_chars, bool(args.drop_nonhangul), bool(args.keep_punct_only)):
                n_skip_filtered += 1
                continue

            src = Path(audio)
            if not src.exists():
                n_skip_missing += 1
                continue

            stem = src.stem
            dst_wav = out_dir / f"{stem}.wav"
            dst_lab = out_dir / f"{stem}.lab"

            # Symlink hoặc copy WAV
            if int(args.symlink):
                try:
                    if dst_wav.exists() or dst_wav.is_symlink():
                        dst_wav.unlink()
                    os.symlink(src.resolve(), dst_wav)
                except OSError:
                    shutil.copy2(src, dst_wav)
            else:
                shutil.copy2(src, dst_wav)

            # Ghi .lab sau khi verbalize
            dst_lab.write_text(text + "\n", encoding="utf-8")
            n_ok += 1

    print(f"Prepared corpus at {out_dir}")
    print(f"Included {n_ok} items.")
    print(f"Verbalized & changed text in {n_verbalized_changed} items.")
    print(f"Skipped {n_skip_missing} rows (missing audio file) và {n_skip_filtered} rows (filtered by text).")

if __name__ == "__main__":
    main()
