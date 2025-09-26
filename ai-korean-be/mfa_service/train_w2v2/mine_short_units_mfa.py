#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, json, argparse, shutil, subprocess, re, unicodedata as ud
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from tqdm import tqdm
from textgrid import TextGrid
from pydub import AudioSegment

HANGUL_SYL_RE = re.compile(r'[\uAC00-\uD7A3]')
NON_HANGUL_RE = re.compile(r'[^\uAC00-\uD7A3\s]')

def load_jsonl(p: Path) -> List[dict]:
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rows.append(json.loads(line))
    return rows

def norm_text(s: str) -> str:
    return ud.normalize("NFC", (s or "").strip())

def count_syllables_ko(s: str) -> int:
    return len([ch for ch in s if HANGUL_SYL_RE.match(ch)])

def strip_to_words(s: str) -> List[str]:
    # Tách từ “thô” theo khoảng trắng, giữ Hangul, bỏ dấu câu khác
    s2 = NON_HANGUL_RE.sub(" ", s)
    toks = [t for t in s2.split() if t]
    return toks

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_wav(src: Path, dst: Path):
    ensure_dir(dst.parent)
    if os.name == "nt":
        # Windows: copy file
        shutil.copy2(src, dst)
    else:
        # *nix: symlink nếu muốn
        try:
            if dst.exists():
                dst.unlink()
            os.symlink(src, dst)
        except Exception:
            shutil.copy2(src, dst)

def write_lab(dst: Path, text: str):
    ensure_dir(dst.parent)
    dst.write_text(text + "\n", encoding="utf-8")

def run_mfa_align(corpus_dir: Path, dict_path: Path, ac_model: Path, out_dir: Path, num_jobs: int = 4):
    # mfa align CORPUS DICT ACOUSTIC OUTDIR
    cmd = [
        "mfa", "align",
        str(corpus_dir), str(dict_path), str(ac_model),
        str(out_dir),
        "--clean", "--overwrite", "--num_jobs", str(num_jobs)
    ]
    print("[MFA] Running:", " ".join(cmd))
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError("MFA align failed")

def load_tg_words_phones(tg_path: Path, tier_words="words", tier_phones="phones"):
    tg = TextGrid.fromFile(str(tg_path))
    words, phones = [], []
    w = next((t for t in tg.tiers if t.name.lower()==tier_words.lower()), None)
    p = next((t for t in tg.tiers if t.name.lower()==tier_phones.lower()), None)
    if w:
        for it in w.intervals:
            txt = (it.mark or "").strip()
            if txt and txt.upper() not in ("SP","SIL","SILENCE","NSN"):
                words.append((int(round(it.minTime*1000)), int(round(it.maxTime*1000)), txt))
    if p:
        for it in p.intervals:
            txt = (it.mark or "").strip()
            if txt:
                phones.append((int(round(it.minTime*1000)), int(round(it.maxTime*1000)), txt))
    return words, phones

def slice_audio(src_wav: Path, start_ms: int, end_ms: int, pad_ms: int, out_dir: Path) -> Tuple[Path,int,int]:
    seg = AudioSegment.from_file(src_wav)
    L = len(seg)
    s = max(0, start_ms - pad_ms)
    e = min(L, end_ms + pad_ms)
    clip = seg[s:e]
    ensure_dir(out_dir)
    out_wav = out_dir / f"{src_wav.stem}_{s:06d}-{e:06d}.wav"
    clip.export(out_wav, format="wav", parameters=["-acodec","pcm_s16le"])
    return out_wav, s, e

def split_long_word_by_phones(word_txt: str, word_s: int, word_e: int, phones: List[Tuple[int,int,str]]) -> List[Tuple[int,int,str]]:
    """Fallback: tách 1 từ nhiều âm tiết thành các cửa sổ 1–2 âm tiết dựa trên phone timing.
       Đơn giản: chia đều theo số âm tiết nếu không map được rõ. """
    syl_count = count_syllables_ko(word_txt)
    if syl_count <= 2:
        return [(word_s, word_e, word_txt)]
    dur = word_e - word_s
    # ước lượng điểm cắt theo tỉ lệ số âm tiết
    per = max(1, dur // syl_count)
    bounds = [word_s]
    for i in range(1, syl_count):
        bounds.append(min(word_e, word_s + i*per))
    bounds.append(word_e)
    # tạo windows 1–2 syllables trượt
    units = []
    i = 0
    while i < syl_count:
        j = min(syl_count, i+2)
        s = bounds[i]
        e = bounds[j]
        units.append((s, e, ""))  # text fill sau ở cấp cao hơn
        i += 2
    return units

def compose_units_from_words(words: List[Tuple[int,int,str]],
                             min_ms=700, max_ms=1500, pad_ms=200) -> List[Tuple[int,int,str]]:
    """Ghép 1–2 âm tiết bằng cách nối 1–n từ liên tiếp sao cho tổng âm tiết ≤ 2 và nằm trong [min_ms, max_ms]."""
    units = []
    i = 0
    n = len(words)
    while i < n:
        s_i, e_i, w_i = words[i]
        syl_i = count_syllables_ko(w_i)
        # nếu từ đơn >2 âm tiết: để fallback xử lý bên ngoài (phones)
        if syl_i == 0:
            i += 1; continue
        # thử 1 từ
        if syl_i <= 2:
            dur1 = e_i - s_i
            if min_ms <= dur1 <= max_ms:
                units.append((s_i, e_i, w_i))
                i += 1
                continue
            # thử ghép với từ kế nếu vẫn ≤2 âm tiết
            if (i+1) < n:
                s_j, e_j, w_j = words[i+1]
                syl_j = count_syllables_ko(w_j)
                if syl_i + syl_j <= 2:
                    dur2 = e_j - s_i
                    if min_ms <= dur2 <= max_ms:
                        units.append((s_i, e_j, f"{w_i} {w_j}"))
                        i += 2
                        continue
        # nếu không phù hợp, nhích cửa sổ
        i += 1
    return units

def main():
    ap = argparse.ArgumentParser(description="Mine 1–2-syllable units from long clips using MFA alignment")
    ap.add_argument("--jsonl", type=str, required=True, help="slices_asr.jsonl (đầu vào)")
    ap.add_argument("--use_text_ref", action="store_true", help="Dùng text_ref để viết .lab (mặc định dùng text)")
    ap.add_argument("--min_source_ms", type=int, default=1500, help="Chỉ align các clip dài hơn ngưỡng này")
    # MFA paths
    ap.add_argument("--run_mfa", action="store_true", help="Chạy MFA align (nếu tắt, chỉ đọc TextGrid đã có)")
    ap.add_argument("--mfa_corpus", type=str, default="mfa_corpus")
    ap.add_argument("--mfa_out", type=str, default="mfa_align")
    ap.add_argument("--mfa_dict", type=str, help="Đường dẫn DICTIONARY (vd: korean_mfa.dict)")
    ap.add_argument("--mfa_acoustic", type=str, help="Đường dẫn ACOUSTIC MODEL (vd: korean_mfa.zip)")
    ap.add_argument("--num_jobs", type=int, default=4)
    ap.add_argument("--tier_words", type=str, default="words")
    ap.add_argument("--tier_phones", type=str, default="phones")
    # Unit mining
    ap.add_argument("--unit_min_ms", type=int, default=700)
    ap.add_argument("--unit_max_ms", type=int, default=1500)
    ap.add_argument("--pad_ms", type=int, default=200)
    ap.add_argument("--export_clips", action="store_true")
    ap.add_argument("--units_out_dir", type=str, default="short_units_wav")
    ap.add_argument("--out_jsonl", type=str, default="short_units.jsonl")
    args = ap.parse_args()

    jsonl_path = Path(args.jsonl)
    rows = load_jsonl(jsonl_path)
    print(f"[INFO] Loaded {len(rows)} items from {jsonl_path}")

    # 1) Chọn các clip dài để align
    long_items = [r for r in rows if int(r.get("duration_ms", 0)) >= args.min_source_ms]
    print(f"[INFO] Selected {len(long_items)} long items (>= {args.min_source_ms} ms) for alignment.")

    corpus_dir = Path(args.mfa_corpus)
    align_out = Path(args.mfa_out)
    ensure_dir(corpus_dir)
    ensure_dir(align_out)

    # 2) Chuẩn bị corpus (.wav + .lab)
    # - Mỗi file là 1 utterance. Tên utt_id = chỉ số + hash gọn.
    mapping = []  # (utt_id, src_wav, transcript_used)
    for idx, r in enumerate(tqdm(long_items, desc="Prepare corpus")):
        wav = Path(r["audio"])
        if not wav.exists():
            print(f"[WARN] missing audio: {wav}")
            continue
        text = norm_text(r.get("text_ref") if args.use_text_ref else r.get("text", ""))
        if not text:
            # fallback nhẹ: dùng text nếu text_ref rỗng
            text = norm_text(r.get("text", ""))
        if not text:
            continue
        utt_id = f"utt{idx:06d}"
        dst_wav = corpus_dir / f"{utt_id}.wav"
        dst_lab = corpus_dir / f"{utt_id}.lab"
        copy_wav(wav, dst_wav)
        write_lab(dst_lab, text)
        mapping.append((utt_id, wav, text))

    print(f"[INFO] Corpus size: {len(mapping)}")

    # 3) MFA align (tùy chọn)
    if args.run_mfa:
        if not (args.mfa_dict and args.mfa_acoustic):
            print("[ERROR] --mfa_dict và --mfa_acoustic là bắt buộc khi --run_mfa")
            sys.exit(1)
        run_mfa_align(corpus_dir, Path(args.mfa_dict), Path(args.mfa_acoustic), align_out, num_jobs=args.num_jobs)

    # 4) Đọc TextGrid & khai thác 1–2 âm tiết
    out_units = []
    units_wav_dir = Path(args.units_out_dir)
    produced = 0

    for utt_id, src_wav, text in tqdm(mapping, desc="Mine units"):
        tg_path = align_out / f"{utt_id}.TextGrid"
        if not tg_path.exists():
            print(f"[WARN] Missing TextGrid for {utt_id}")
            continue
        words, phones = load_tg_words_phones(tg_path, args.tier_words, args.tier_phones)
        if not words:
            continue

        # Tạo units từ từ (≤2 âm tiết)
        units = compose_units_from_words(words, min_ms=args.unit_min_ms, max_ms=args.unit_max_ms, pad_ms=args.pad_ms)

        # Fallback: từ đơn >2 âm tiết → tách bằng phones (xấp xỉ)
        # Tìm từ nào dài >2 âm tiết mà bị bỏ sót
        for (s, e, wtxt) in words:
            if count_syllables_ko(wtxt) > 2:
                # kiểm tra xem có unit nào đã cover vùng này chưa
                covered = any(not (e <= us or s >= ue) for (us, ue, _) in units)
                if not covered:
                    for (ss, ee, _) in split_long_word_by_phones(wtxt, s, e, phones):
                        dur = ee - ss
                        if args.unit_min_ms <= dur <= args.unit_max_ms:
                            units.append((ss, ee, wtxt))

        # Xuất
        for (s, e, utext) in units:
            if args.export_clips:
                out_wav, s2, e2 = slice_audio(src_wav, s, e, args.pad_ms, units_wav_dir)
            else:
                out_wav, s2, e2 = src_wav, s, e

            out_units.append({
                "audio": str(out_wav),
                "start_ms": s2,
                "end_ms": e2,
                "duration_ms": e2 - s2,
                "text": utext,              # chính tả (theo words tier)
                "text_ref": None,           # bạn có thể post-process để sinh surface/ref
                "source": str(src_wav),
                "kind": "unit_1_2_syll",
                "meta": {
                    "from": "MFA",
                    "tier_words": args.tier_words,
                    "tier_phones": args.tier_phones,
                    "pad_ms": args.pad_ms
                }
            })
            produced += 1

    # 5) Ghi JSONL
    out_jsonl = Path(args.out_jsonl)
    ensure_dir(out_jsonl.parent)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for o in out_units:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    print(f"[DONE] Units produced: {produced}")
    print(f"- JSONL: {out_jsonl.resolve()}")
    if args.export_clips:
        print(f"- WAV out: {units_wav_dir.resolve()}")
    print(f"- TextGrids used: {align_out.resolve()}")
    print("Tip: bạn có thể post-process để thêm text_ref/jamos_ref/phonemes_ref cho từng unit.")

if __name__ == "__main__":
    main()


# python mine_short_units_mfa.py ^
#   --jsonl "E:\USE\My_project\source audio\output_vad\slices_asr.jsonl" ^
#   --run_mfa ^
#   --mfa_corpus "E:\USE\My_project\mfa_corpus" ^
#   --mfa_out    "E:\USE\My_project\mfa_align" ^
#   --mfa_dict   "E:\MFA\korean_mfa.dict" ^
#   --mfa_acoustic "E:\MFA\korean_mfa.zip" ^
#   --use_text_ref ^
#   --min_source_ms 1500 ^
#   --unit_min_ms 700 ^
#   --unit_max_ms 1500 ^
#   --pad_ms 200 ^
#   --export_clips ^
#   --units_out_dir "E:\USE\My_project\short_units_wav" ^
#   --out_jsonl "E:\USE\My_project\short_units.jsonl"
