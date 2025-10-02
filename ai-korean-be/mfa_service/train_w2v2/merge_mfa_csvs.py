# merge_mfa_csvs.py
# Gộp các CSV lẻ của MFA thành 2 file tổng: aligned_words.csv, aligned_phones.csv

import csv, glob, os, pathlib, sys

# ==== Sửa 2 biến này theo đường dẫn của bạn ====
ALIGN_DIR = r"E:/USE/My_project/source audio/out_align"      # thư mục chứa các CSV theo từng audio
OUT_DIR   = r"E:/USE/My_project/source audio/out_align_csv"  # nơi ghi 2 file gộp
# ===============================================

pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
words_out  = os.path.join(OUT_DIR, "aligned_words.csv")
phones_out = os.path.join(OUT_DIR, "aligned_phones.csv")

def merge(kind: str, out_path: str):
    # kind in {"words","phones"}; tìm mọi file *_words.csv hoặc *_phones.csv
    pattern = os.path.join(ALIGN_DIR, "**", f"*{kind}.csv")
    files = glob.glob(pattern, recursive=True)
    if not files:
        print(f"[WARN] Không tìm thấy file {kind}.csv trong {ALIGN_DIR}")
        return
    writer = None
    header = None
    total = 0
    with open(out_path, "w", encoding="utf-8", newline="") as fout:
        for fp in files:
            with open(fp, "r", encoding="utf-8", newline="") as fin:
                reader = csv.DictReader(fin)
                rows = list(reader)
                if not rows:
                    continue
                # tên stem từ file (bỏ hậu tố _words/_phones)
                stem = pathlib.Path(fp).stem
                if stem.endswith(f"_{kind}"):
                    stem = stem[:-(len(kind)+1)]
                # đảm bảo có cột 'file'
                for r in rows:
                    if "file" not in r or not r["file"]:
                        r["file"] = stem
                # chuẩn hoá header và ghi
                if writer is None:
                    header = list(rows[0].keys())
                    if "file" not in header:
                        header.append("file")
                    writer = csv.DictWriter(fout, fieldnames=header)
                    writer.writeheader()
                for r in rows:
                    for h in header:
                        r.setdefault(h, "")
                    writer.writerow(r)
                    total += 1
    print(f"[OK] Gộp {kind}: {total} dòng -> {out_path}")

merge("words",  words_out)
merge("phones", phones_out)
print("Done.")
