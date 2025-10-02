#!/usr/bin/env python
# batch_splice_from_align.py
# Batch splice ALL files from MFA CSV outputs into kaggle/working.
# It calls splice_by_phones_ko.py per file.
#
# Usage:
#   python batch_splice_from_align.py \
#       --words_csv /kaggle/working/out_align/aligned_words.csv \
#       --phones_csv /kaggle/working/out_align/aligned_phones.csv \
#       --audio_root /kaggle/tmp/corpus \
#       --splicer /kaggle/working/splice_by_phones_ko.py \
#       --min_ms 400 --max_ms 700 --pad_pre_ms 15 --pad_post_ms 30 \
#       --write_wavs 1 \
#       --drop_empty 1 --drop_nonhangul 1 --min_chars 1 --keep_punct_only 0
import csv, argparse, subprocess, shlex
from pathlib import Path

def read_csv(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words_csv", required=True)
    ap.add_argument("--phones_csv", required=True)
    ap.add_argument("--audio_root", default="/kaggle/tmp/corpus")
    ap.add_argument("--splicer", default="/kaggle/working/splice_by_phones_ko.py")
    ap.add_argument("--out_root", default="/kaggle/working")
    ap.add_argument("--min_ms", type=int, default=400)
    ap.add_argument("--max_ms", type=int, default=700)
    ap.add_argument("--pad_pre_ms", type=int, default=15)
    ap.add_argument("--pad_post_ms", type=int, default=30)
    ap.add_argument("--write_wavs", type=int, default=1)
    ap.add_argument("--drop_empty", type=int, default=1)
    ap.add_argument("--drop_nonhangul", type=int, default=1)
    ap.add_argument("--min_chars", type=int, default=1)
    ap.add_argument("--keep_punct_only", type=int, default=0)
    args = ap.parse_args()

    words = read_csv(Path(args.words_csv))
    phones = read_csv(Path(args.phones_csv))

    # Collect unique file stems from phones CSV (MFA usually stores in 'file')
    stems = []
    seen = set()
    for row in phones:
        stem = row.get("file") or row.get("utterance") or row.get("filename")
        if stem and stem not in seen:
            stems.append(stem)
            seen.add(stem)

    print(f"Found {len(stems)} files in alignment.")

    for stem in stems:
        wav = Path(args.audio_root) / f"{stem}.wav"
        if not wav.exists():
            print(f"[WARN] Missing audio for {stem}: {wav}")
            continue

        out_dir = Path(args.out_root) / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python", args.splicer,
            "--audio", str(wav),
            "--words_csv", args.words_csv,
            "--phones_csv", args.phones_csv,
            "--out_dir", str(out_dir),
            "--filename_key", stem,
            "--min_ms", str(args.min_ms),
            "--max_ms", str(args.max_ms),
            "--pad_pre_ms", str(args.pad_pre_ms),
            "--pad_post_ms", str(args.pad_post_ms),
            "--write_wavs", str(args.write_wavs),
            "--drop_empty", str(args.drop_empty),
            "--drop_nonhangul", str(args.drop_nonhangul),
            "--min_chars", str(args.min_chars),
            "--keep_punct_only", str(args.keep_punct_only),
        ]
        print("Running:", " ".join(shlex.quote(x) for x in cmd))
        subprocess.run(cmd, check=True)

    print("Batch splicing done. Outputs are under", args.out_root)

if __name__ == "__main__":
    main()
