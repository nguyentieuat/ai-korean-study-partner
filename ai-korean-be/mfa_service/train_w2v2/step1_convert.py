# step1_convert.py
import argparse, os, sys, subprocess, pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed

AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".aac", ".wma", ".mp2", ".opus"}

def run_ffmpeg(inp, outp, overwrite=False):
    outp.parent.mkdir(parents=True, exist_ok=True)
    if outp.exists() and not overwrite:
        return "skip"
    cmd = [
        "ffmpeg", "-y" if overwrite else "-n",
        "-i", str(inp),
        "-ac", "1", "-ar", "16000", "-vn", "-map_metadata", "-1",
        str(outp),
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return "ok"
    except subprocess.CalledProcessError as e:
        return f"err:{e.returncode}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Thư mục chứa audio gốc")
    ap.add_argument("--out", dest="out", required=True, help="Thư mục output WAV 16k")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    in_root = pathlib.Path(args.inp).resolve()
    out_root = pathlib.Path(args.out).resolve()

    files = [p for p in in_root.rglob("*") if p.suffix.lower() in AUDIO_EXTS]
    if not files:
        print("Không tìm thấy file audio.")
        sys.exit(1)

    print(f"Found {len(files)} files. Converting to 16k mono…")

    futs = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for src in files:
            rel = src.relative_to(in_root)
            outp = out_root / rel.with_suffix(".16k.wav")
            futs.append(ex.submit(run_ffmpeg, src, outp, args.overwrite))

        done = 0
        for f in as_completed(futs):
            done += 1
            if done % 50 == 0:
                print(f"Progress: {done}/{len(futs)}")

    print("Done.")

if __name__ == "__main__":
    main()


# python train_w2v2/step1_convert.py --in "E:\USE\My_project\source audio\audio train w2v2" --out "E:\USE\My_project\source audio\audio train w2v2_16k" --workers 8