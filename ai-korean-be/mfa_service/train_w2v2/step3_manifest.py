# step3_manifest.py
import argparse, pathlib, csv, re
import soundfile as sf

COLUMNS = [
    "source_path", "start_ms", "end_ms", "slice_path", "duration_ms",
    "sil_min_len", "sil_offset_db", "sil_keep_ms", "sil_join_ms", "max_ms",
    "asr_text", "text_ref", "jamos_ref", "phonemes_ref"
]

NAME_RE = re.compile(r"_(\d{6})-(\d{6})\.wav$", re.IGNORECASE)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sliced", required=True, help="Thư mục chứa clip đã cắt (từ step2)")
    ap.add_argument("--out_csv", required=True, help="Đường dẫn manifest.csv")
    ap.add_argument("--sil_min_len", type=int, default=240)
    ap.add_argument("--sil_offset_db", type=float, default=8.0)
    ap.add_argument("--sil_keep_ms", type=int, default=180)
    ap.add_argument("--sil_join_ms", type=int, default=200)
    ap.add_argument("--max_ms", type=int, default=15000)
    args = ap.parse_args()

    root = pathlib.Path(args.sliced).resolve()
    wavs = list(root.rglob("*.wav"))
    if not wavs:
        print("Không thấy clip nào. Đảm bảo đã chạy step2.")
        return

    rows = []
    for p in wavs:
        try:
            info = sf.info(str(p))
            dur_ms = int(round(1000 * info.frames / info.samplerate))
        except Exception:
            continue

        source_path = ""  # nếu muốn, bạn có thể nhúng đường dẫn nguồn gốc ở đây
        start_ms = ""
        end_ms = ""
        m = NAME_RE.search(p.name)
        if m:
            start_ms, end_ms = int(m.group(1)), int(m.group(2))

        rows.append({
            "source_path": source_path,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "slice_path": str(p),
            "duration_ms": dur_ms,
            "sil_min_len": args.sil_min_len,
            "sil_offset_db": args.sil_offset_db,
            "sil_keep_ms": args.sil_keep_ms,
            "sil_join_ms": args.sil_join_ms,
            "max_ms": args.max_ms,
            "asr_text": "",
            "text_ref": "",
            "jamos_ref": "",
            "phonemes_ref": "",
        })

    out_csv = pathlib.Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_csv}")

if __name__ == "__main__":
    main()


# python train_w2v2/step3_manifest.py --sliced "E:\USE\My_project\source audio\output_vad" --out_csv ""E:\USE\My_project\source audio\manifest.csv"