#!/usr/bin/env python
# batch_splice_from_align_dir.py
# Đọc CSV MFA per-audio, gom theo stem, verbalize số+đơn vị cho text_ref,
# gọi splicer, và (tuỳ chọn) gộp manifest tổng.
#
# Example (PowerShell):
# python train_w2v2/batch_splice_from_align_dir.py `
#   --csv_dir "E:/USE/My_project/source audio/out_align" `
#   --audio_root "E:/USE/My_project/source audio/corpus" `
#   --splicer train_w2v2/splice_by_phones_ko.py `
#   --out_root "E:/USE/My_project/out_splices" `
#   --global_manifest "E:/USE/My_project/out_splices/dataset.jsonl" `
#   --manifest_format jsonl `
#   --min_ms 400 --max_ms 700 --pad_pre_ms 15 --pad_post_ms 30 `
#   --write_wavs 1 --drop_empty 1 --drop_nonhangul 1 --min_chars 1 --keep_punct_only 0 `
#   --verbalize_numbers 1 --verbalize_mode auto --log_level INFO
import argparse, csv, os, tempfile, subprocess, shlex, sys, json, re, logging, time
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple, Optional

# ============== Logging ==============
def setup_logger(level: str = "INFO"):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

# ============== Number verbalizer (KO) ==============
_SINO_DIG   = ["영","일","이","삼","사","오","육","칠","팔","구"]
_SINO_SMALL = ["","십","백","천"]
_SINO_BIG   = [("조",10**12),("억",10**8),("만",10**4)]

_NATIVE_ONES = {
    1:"한", 2:"두", 3:"세", 4:"네", 5:"다섯", 6:"여섯", 7:"일곱", 8:"여덟", 9:"아홉",
    10:"열", 20:"스물", 30:"서른", 40:"마흔", 50:"쉰", 60:"예순", 70:"일흔", 80:"여든", 90:"아흔"
}

def _native_number(n: int) -> str:
    if n <= 0: return "영"
    if n in _NATIVE_ONES: return _NATIVE_ONES[n]
    tens = (n//10)*10
    ones = n%10
    return f"{_NATIVE_ONES.get(tens, '')}{_NATIVE_ONES.get(ones, '')}"

def read_number_native(n: int) -> str:
    if n == 20: return "스무"
    return _native_number(n)

def _read_chunk_4(n: int) -> str:
    s = []
    for i in range(3,-1,-1):
        unit = 10**i
        d = (n // unit) % 10
        if d == 0: continue
        if i >= 1:
            if d == 1: s.append(_SINO_SMALL[i])
            else: s.append(_SINO_DIG[d] + _SINO_SMALL[i])
        else:
            s.append(_SINO_DIG[d])
    return "".join(s) if s else ""

def read_number_sino(num_str: str) -> str:
    s = num_str.replace(",", "")
    if s.count(".") > 1: return s
    if "." in s:
        int_part, frac_part = s.split(".", 1)
    else:
        int_part, frac_part = s, ""
    try:
        n = int(int_part) if int_part else 0
    except Exception:
        return num_str
    if n == 0:
        int_read = "영"
    else:
        parts = []; rest = n
        for big_name, big_val in _SINO_BIG:
            q = rest // big_val
            if q:
                parts.append(_read_chunk_4(q) + big_name); rest %= big_val
        small = _read_chunk_4(rest)
        if small: parts.append(small)
        int_read = "".join(parts) if parts else "영"
    if frac_part:
        frac_read = "점 " + " ".join(_SINO_DIG[int(d)] if d.isdigit() else d for d in frac_part)
        return f"{int_read} {frac_read}"
    return int_read

# Units/counters
_UNIT_MAP: Dict[str, Tuple[str,str]] = {
    # metric
    "MM": ("밀리미터", "sino"),
    "CM": ("센티미터", "sino"),
    "KM": ("킬로미터", "sino"),
    "KG": ("킬로그램", "sino"),
    "G":  ("그램", "sino"),
    "T":  ("톤", "sino"),
    "L":  ("리터", "sino"),
    "ML": ("밀리리터", "sino"),
    "ΜL": ("마이크로리터", "sino"),  # Greek mu
    "UL": ("마이크로리터", "sino"),
    # time
    "H":  ("시간", "native"),
    "HR": ("시간", "native"),
    "MIN":("분", "sino"),
    "S":  ("초", "sino"),
    "MS": ("밀리초", "sino"),
    # percent & temp & currency
    "%":  ("퍼센트", "sino"),
    "°C": ("도씨", "sino"),
    "℃":  ("도씨", "sino"),
    "KRW":("원", "sino"),
}

_NATIVE_COUNTERS = {"개","명","살","시","마리","번","권","대","잔","병","송이","켤레","장","과","가지"}

# Regex skeletons
_num_core = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_num_unit_pat = re.compile(
    rf"(?P<num>{_num_core})\s*(?P<unit>%|℃|°C|mm|cm|km|kg|g|t|l|ml|μl|ul|h|hr|min|s|ms|KRW)",
    re.IGNORECASE,
)
_currency_pat = re.compile(
    rf"(?P<cur>₩|원|KRW)\s*(?P<num>{_num_core})",
    re.IGNORECASE,
)
# _counter_pat will be built at runtime after knowing counters:
_counter_pat: Optional[re.Pattern] = None

_time_colon_pat = re.compile(
    r"(?P<h>\d{1,2})\s*:\s*(?P<m>\d{2})(?:\s*:\s*(?P<s>\d{2}))?"
)
_date_pat = re.compile(
    r"(?P<y>\d{4})\s*[-/.]\s*(?P<m>\d{1,2})\s*[-/.]\s*(?P<d>\d{1,2})"
)
_phone_pat = re.compile(
    r"\b(?P<num>(?:\d-?){7,})\b"
)

def init_verbalizer():
    """Build the counter regex using current counter set."""
    global _counter_pat
    ctr_union = "|".join(map(re.escape, _NATIVE_COUNTERS))
    _counter_pat = re.compile(
        rf"(?P<num>{_num_core})\s*(?P<ctr>{ctr_union})"
    )

def verbalize_numbers_ko(text: str, mode: str = "auto") -> str:
    """
    mode: auto|sino|native
    Order: phone → date → hh:mm(:ss) → currency → number+unit → counters → bare numbers
    """
    assert _counter_pat is not None, "Call init_verbalizer() first."

    def _read_digits(s: str) -> str:
        m = re.sub(r"[^\d]", "", s)
        DIG = ["공","일","이","삼","사","오","육","칠","팔","구"]
        return " ".join(DIG[int(ch)] for ch in m)

    # phones / codes
    text = _phone_pat.sub(lambda m: _read_digits(m.group("num")), text)

    # date
    def _date_repl(m: re.Match) -> str:
        y = read_number_sino(m.group("y")) + "년"
        mo = read_number_sino(m.group("m")) + "월"
        d = read_number_sino(m.group("d")) + "일"
        return f"{y} {mo} {d}"
    text = _date_pat.sub(_date_repl, text)

    # time hh:mm(:ss)
    def _time_repl(m: re.Match) -> str:
        h = int(m.group("h")); mm = int(m.group("m"))
        ss = int(m.group("s")) if m.group("s") else None
        h_read = read_number_native(h) + " 시"
        m_read = (read_number_sino(str(mm)) + " 분") if mm else ""
        s_read = (read_number_sino(str(ss)) + " 초") if ss is not None and ss else ""
        return " ".join(x for x in [h_read, m_read, s_read] if x)
    text = _time_colon_pat.sub(_time_repl, text)

    # currency
    def _cur_repl(m: re.Match) -> str:
        num = read_number_sino(m.group("num"))
        return f"{num} 원"
    text = _currency_pat.sub(_cur_repl, text)

    # number + unit
    def _unit_repl(m: re.Match) -> str:
        raw_unit = m.group("unit")
        unit = raw_unit.upper()
        # normalize special 'l' to uppercase L
        if unit == "L" or unit == "ML" or unit == "UL":
            pass
        num = m.group("num")
        unit_word, u_mode = _UNIT_MAP.get(unit, ("", "sino"))
        if u_mode == "native":
            try:
                spoken = read_number_native(int(num.replace(",","").split(".")[0]))
            except:
                spoken = read_number_sino(num)
        else:
            spoken = read_number_sino(num)
        return f"{spoken} {unit_word}".strip()
    text = _num_unit_pat.sub(_unit_repl, text)

    # counters (native)
    def _ctr_repl(m: re.Match) -> str:
        num = m.group("num")
        ctr = m.group("ctr")
        try:
            n = int(num.replace(",","").split(".")[0])
            head = read_number_native(n)
        except:
            head = read_number_sino(num)
        return f"{head} {ctr}"
    text = _counter_pat.sub(_ctr_repl, text)

    # bare numbers → sino
    def _solo_repl(m: re.Match) -> str:
        return read_number_sino(m.group(0))
    text = re.sub(rf"\b{_num_core}\b", _solo_repl, text)

    return text

# ============== CSV helpers ==============
def _lower_map(fieldnames):
    return {c.lower(): c for c in (fieldnames or [])}

def _get(row, cmap, *keys):
    for k in keys:
        if k in cmap:
            return row.get(cmap[k], "")
    return ""

def _stem_from_filename(fp: Path):
    stem = fp.stem
    for suf in ("_words", "_word", "_phones", "_phone"):
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem

# ============== Collect per stem ==============
def collect_by_stem(csv_dir: Path):
    data = defaultdict(lambda: {"words": [], "phones": []})
    files = sorted(csv_dir.rglob("*.csv"))
    if not files:
        logging.error(f"No CSV files found under: {csv_dir}")
        return data

    total_rows = 0
    for fp in files:
        try:
            with fp.open("r", encoding="utf-8", newline="") as fin:
                reader = csv.DictReader(fin)
                cmap = _lower_map(reader.fieldnames)
                stem = _stem_from_filename(fp)
                for row in reader:
                    total_rows += 1
                    typ   = (_get(row, cmap, "type") or "").strip().lower()
                    begin = _get(row, cmap, "begin", "start")
                    end   = _get(row, cmap, "end", "finish")
                    label = _get(row, cmap, "label", "text", "value")
                    if not begin or not end:
                        continue
                    if typ in ("word", "words"):
                        data[stem]["words"].append({"file": stem,"begin": begin,"end": end,"word": label or ""})
                    elif typ in ("phone", "phones", "phoneme", "phonemes"):
                        data[stem]["phones"].append({"file": stem,"begin": begin,"end": end,"phone": (label or "").strip()})
                    else:
                        lowname = fp.name.lower()
                        if "phone" in lowname:
                            data[stem]["phones"].append({"file": stem,"begin": begin,"end": end,"phone": (label or "").strip()})
                        elif "word" in lowname:
                            data[stem]["words"].append({"file": stem,"begin": begin,"end": end,"word": label or ""})
        except UnicodeDecodeError:
            with fp.open("r", encoding="utf-8-sig", newline="") as fin:
                reader = csv.DictReader(fin)
                cmap = _lower_map(reader.fieldnames)
                stem = _stem_from_filename(fp)
                for row in reader:
                    total_rows += 1
                    typ   = (_get(row, cmap, "type") or "").strip().lower()
                    begin = _get(row, cmap, "begin", "start")
                    end   = _get(row, cmap, "end", "finish")
                    label = _get(row, cmap, "label", "text", "value")
                    if not begin or not end:
                        continue
                    if typ == "word":
                        data[stem]["words"].append({"file": stem,"begin": begin,"end": end,"word": label or ""})
                    elif typ in ("phone","phoneme"):
                        data[stem]["phones"].append({"file": stem,"begin": begin,"end": end,"phone": (label or "").strip()})
                    else:
                        lowname = fp.name.lower()
                        if "phone" in lowname:
                            data[stem]["phones"].append({"file": stem,"begin": begin,"end": end,"phone": (label or "").strip()})
                        elif "word" in lowname:
                            data[stem]["words"].append({"file": stem,"begin": begin,"end": end,"word": label or ""})

    logging.info(f"Scanned {len(files)} CSV files (~{total_rows} rows).")
    return data

# ============== Temp CSV writers ==============
def write_temp_csv(rows, kind: str, verbalize: bool, vmode: str):
    """kind: 'words' or 'phones'. If words+verbalize, apply verbalizer."""
    suffix = "_words.csv" if kind == "words" else "_phones.csv"
    tmp = tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, suffix=suffix)
    path = tmp.name
    header = ["file","begin","end","word"] if kind == "words" else ["file","begin","end","phone"]
    writer = csv.DictWriter(tmp, fieldnames=header)
    writer.writeheader()

    changed = 0
    for r in rows:
        out = {k: r.get(k, "") for k in header}
        if kind == "words" and verbalize:
            w0 = out["word"]
            w1 = verbalize_numbers_ko(w0, mode=vmode)
            if w1 != w0:
                changed += 1
            out["word"] = w1
        writer.writerow(out)
    tmp.close()
    return path, changed

# ============== MAIN ==============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", required=True)
    parser.add_argument("--audio_root", required=True)
    parser.add_argument("--splicer", required=True)
    parser.add_argument("--out_root", required=True)
    # pass-through
    parser.add_argument("--min_ms", type=int, default=400)
    parser.add_argument("--max_ms", type=int, default=700)
    parser.add_argument("--pad_pre_ms", type=int, default=15)
    parser.add_argument("--pad_post_ms", type=int, default=30)
    parser.add_argument("--write_wavs", type=int, default=1)
    parser.add_argument("--drop_empty", type=int, default=1)
    parser.add_argument("--drop_nonhangul", type=int, default=1)
    parser.add_argument("--min_chars", type=int, default=1)
    parser.add_argument("--keep_punct_only", type=int, default=0)
    # global manifest
    parser.add_argument("--global_manifest", default="")
    parser.add_argument("--manifest_format", default="jsonl", choices=["jsonl","csv"])
    # verbalize
    parser.add_argument("--verbalize_numbers", type=int, default=1)
    parser.add_argument("--verbalize_mode", default="auto", choices=["auto","sino","native"])
    # logging
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()

    setup_logger(args.log_level)
    init_verbalizer()

    csv_dir = Path(args.csv_dir)
    audio_root = Path(args.audio_root)
    splicer = Path(args.splicer)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    logging.info("Collecting CSV…")
    data = collect_by_stem(csv_dir)
    stems = sorted(k for k,v in data.items() if v["phones"])
    logging.info(f"{len(stems)} stems with phones found.")

    # global manifest
    gm_path = Path(args.global_manifest) if args.global_manifest else None
    gm = None
    csv_writer = None
    if gm_path:
        gm_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if gm_path.exists() else "w"
        gm = open(gm_path, mode, encoding="utf-8", newline="")
        if args.manifest_format == "csv" and mode == "w":
            import csv as _csv
            header = ["file","index","audio","t_begin","t_end","duration","speaker",
                      "text","text_ref","jamos","jamos_ref","phonemes","phonemes_ref","n_phones","phones","rule_tags"]
            csv_writer = _csv.DictWriter(gm, fieldnames=header)
            csv_writer.writeheader()
        logging.info(f"Global manifest → {gm_path} ({args.manifest_format})")

    t0 = time.time()
    total_segments = 0
    processed = 0

    for i, stem in enumerate(stems, 1):
        wav = audio_root / f"{stem}.wav"
        if not wav.exists():
            logging.warning(f"[{i}/{len(stems)}] Missing audio for {stem}: {wav}")
            continue

        words = data[stem]["words"]
        phones = data[stem]["phones"]
        if not phones:
            logging.warning(f"[{i}/{len(stems)}] No phones for {stem}, skip.")
            continue

        tmp_words, changed = write_temp_csv(words, "words", bool(args.verbalize_numbers), args.verbalize_mode)
        tmp_phones, _ = write_temp_csv(phones, "phones", False, args.verbalize_mode)

        out_dir = out_root / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"[{i}/{len(stems)}] {stem}: words={len(words)}, phones={len(phones)}, verbalized_changed={changed}")

        cmd = [
            sys.executable, str(splicer),
            "--audio", str(wav),
            "--words_csv", tmp_words,
            "--phones_csv", tmp_phones,
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
        logging.debug("[RUN] " + " ".join(shlex.quote(c) for c in cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"[{stem}] Splicer failed: {e}")
        finally:
            try: os.remove(tmp_words)
            except: pass
            try: os.remove(tmp_phones)
            except: pass

        # merge to global
        seg_count = 0
        manifest_path = out_dir / "splices.jsonl"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as mf:
                for line in mf:
                    if not line.strip():
                        continue
                    seg_count += 1
                    if gm:
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        if rec.get("audio"):
                            rec["audio"] = str((out_dir / rec["audio"]).resolve())
                        if args.manifest_format == "jsonl":
                            gm.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        else:
                            row = {
                                "file": rec.get("file",""),
                                "index": rec.get("index",""),
                                "audio": rec.get("audio",""),
                                "t_begin": rec.get("t_begin",""),
                                "t_end": rec.get("t_end",""),
                                "duration": rec.get("duration",""),
                                "speaker": rec.get("speaker",""),
                                "text": rec.get("text",""),
                                "text_ref": rec.get("text_ref",""),
                                "jamos": rec.get("jamos",""),
                                "jamos_ref": rec.get("jamos_ref",""),
                                "phonemes": rec.get("phonemes",""),
                                "phonemes_ref": rec.get("phonemes_ref",""),
                                "n_phones": rec.get("n_phones",""),
                                "phones": " ".join(rec.get("phones",[])) if isinstance(rec.get("phones"), list) else rec.get("phones",""),
                                "rule_tags": " ".join(rec.get("rule_tags",[])) if isinstance(rec.get("rule_tags"), list) else rec.get("rule_tags",""),
                            }
                            if csv_writer is None:
                                import csv as _csv
                                header = list(row.keys())
                                csv_writer = _csv.DictWriter(gm, fieldnames=header)
                                csv_writer.writeheader()
                            csv_writer.writerow(row)

        total_segments += seg_count
        processed += 1
        logging.info(f"[{i}/{len(stems)}] {stem}: segments={seg_count}")

    if gm: gm.close()
    dt = time.time() - t0
    logging.info(f"[DONE] Stems processed={processed}/{len(stems)}, total segments={total_segments}, time={dt:.1f}s")
    if gm_path:
        logging.info(f"[DONE] Global manifest: {gm_path}")

if __name__ == "__main__":
    main()
