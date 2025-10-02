#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv, json, argparse, subprocess, shlex, re
from pathlib import Path
from typing import List, Dict, Any, Tuple

# ---------- Optional ko_phonology (text -> surface) ----------
def surface_from_text(text: str) -> str:
    try:
        import ko_phonology  # type: ignore
        if hasattr(ko_phonology, "to_surface"):
            return ko_phonology.to_surface(text)
        if hasattr(ko_phonology, "explain_clean_for_user"):
            surf, _ = ko_phonology.explain_clean_for_user(text)
            return surf
    except Exception:
        pass
    return text

# ---------- Optional ko_phonology_ipa (phones -> surface phones) ----------
def surface_from_ipa_list(phones: List[str]) -> Tuple[List[str], List[str]]:
    try:
        import ko_phonology_ipa  # type: ignore
        if hasattr(ko_phonology_ipa, "apply_korean_phonology"):
            return ko_phonology_ipa.apply_korean_phonology(phones)  # type: ignore
    except Exception:
        pass
    return phones, []

# ---------- Hangul decomposition to jamo (no deps) ----------
CHO = [u"ᄀ",u"ᄁ",u"ᄂ",u"ᄃ",u"ᄄ",u"ᄅ",u"ᄆ",u"ᄇ",u"ᄈ",u"ᄉ",u"ᄊ",u"ᄋ",u"ᄌ",u"ᄍ",u"ᄎ",u"ᄏ",u"ᄐ",u"ᄑ",u"ᄒ"]
JUNG = [u"ᅡ",u"ᅢ",u"ᅣ",u"ᅤ",u"ᅥ",u"ᅦ",u"ᅧ",u"ᅨ",u"ᅩ",u"ᅪ",u"ᅫ",u"ᅬ",u"ᅭ",u"ᅮ",u"ᅯ",u"ᅰ",u"ᅱ",u"ᅲ",u"ᅳ",u"ᅴ",u"ᅵ"]
JONG = [""] + [u"ᆨ",u"ᆩ",u"ᆪ",u"ᆫ",u"ᆬ",u"ᆭ",u"ᆮ",u"ᆯ",u"ᆰ",u"ᆱ",u"ᆲ",u"ᆳ",u"ᆴ",u"ᆵ",u"ᆶ",u"ᆷ",u"ᆸ",u"ᆹ",u"ᆺ",u"ᆻ",u"ᆼ",u"ᆽ",u"ᆾ",u"ᆿ",u"ᇀ",u"ᇁ",u"ᇂ"]

def hangul_to_jamo(s: str) -> str:
    res = []
    for ch in s:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            base = code - 0xAC00
            cho = base // 588
            jung = (base % 588) // 28
            jong = base % 28
            res.extend([CHO[cho], JUNG[jung]])
            if JONG[jong]:
                res.append(JONG[jong])
        else:
            res.append(ch)
    return "".join(res)

# ---------- CSV helpers ----------
def read_csv(path: Path):
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [row for row in r]

def to_f(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0

# ---------- Simple text quality filters ----------
_HANGUL_RE = re.compile(r"[\uAC00-\uD7A3]")
_PUNCT_ONLY_RE = re.compile(r"^[\s\.,?!;:\"'()\[\]{}~\-–—…·]+$")

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

# ---------- Vowel set (IPA-ish) to prefer splits at syllable boundaries ----------
KOR_VOWELS = set(["a","e","i","o","u","ɯ","ʌ","ɛ","ɐ","ə","ʊ","ø","y","ɨ","ɤ","œ","ɜ","æ","ʉ","ɒ","j","w"])

def splice_phones(phones, min_ms: int, max_ms: int):
    segs = []
    i, n = 0, len(phones)
    while i < n:
        j = i
        start = to_f(phones[i]["begin"])
        while j < n:
            dur_ms = int(1000 * (to_f(phones[j]["end"]) - start))
            if dur_ms >= max_ms:
                # try to back off to vowel boundary j
                k = j
                while k > i+1:
                    ph = phones[k]["phone"]
                    prev = phones[k-1]["phone"]
                    if (ph in KOR_VOWELS) and (prev not in KOR_VOWELS):
                        j = k
                        break
                    k -= 1
                break
            j += 1
        if j == i:
            j = min(i+1, n)
        while j < n and int(1000 * (to_f(phones[j-1]["end"]) - start)) < min_ms:
            j += 1
        segs.append((i, j))
        i = j
    return segs

def ffmpeg_cut(src: Path, dst: Path, t0: float, t1: float, pad_pre_ms: int, pad_post_ms: int):
    dst.parent.mkdir(parents=True, exist_ok=True)
    t0p = max(0.0, t0 - pad_pre_ms/1000.0)
    t1p = max(t0p, t1 + pad_post_ms/1000.0)
    dur = max(0.0, t1p - t0p)
    cmd = f'ffmpeg -ss {t0p:.3f} -i {shlex.quote(str(src))} -t {dur:.3f} -acodec pcm_s16le -ar 16000 -ac 1 -y {shlex.quote(str(dst))}'
    subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--words_csv", required=True)
    ap.add_argument("--phones_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--filename_key", default=None)
    ap.add_argument("--speaker", default=None)
    ap.add_argument("--min_ms", type=int, default=400)
    ap.add_argument("--max_ms", type=int, default=700)
    ap.add_argument("--pad_pre_ms", type=int, default=15)
    ap.add_argument("--pad_post_ms", type=int, default=30)
    ap.add_argument("--write_wavs", type=int, default=0)
    # new filters
    ap.add_argument("--drop_empty", type=int, default=1)
    ap.add_argument("--drop_nonhangul", type=int, default=1)
    ap.add_argument("--min_chars", type=int, default=1)
    ap.add_argument("--keep_punct_only", type=int, default=0)
    args = ap.parse_args()

    audio = Path(args.audio).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_wavs = out_dir / "splices"
    out_manifest = out_dir / "splices.jsonl"

    words = read_csv(Path(args.words_csv))
    phones = read_csv(Path(args.phones_csv))

    key = args.filename_key or audio.stem
    if args.speaker:
        words = [w for w in words if w.get("file")==key and w.get("speaker")==args.speaker and w.get("word","").strip()!=""]
        phones = [p for p in phones if p.get("file")==key and p.get("speaker")==args.speaker]
    else:
        words = [w for w in words if w.get("file")==key and w.get("word","").strip()!=""]
        phones = [p for p in phones if p.get("file")==key]

    # Build splices
    spans = splice_phones(phones, args.min_ms, args.max_ms)

    kept = 0
    dropped = 0
    with out_manifest.open("w", encoding="utf-8") as f:
        for n,(si, sj) in enumerate(spans):
            ph_chunk = phones[si:sj]
            t0 = float(ph_chunk[0]["begin"])
            t1 = float(ph_chunk[-1]["end"])
            ph_list = [p["phone"] for p in ph_chunk]
            ph_str = " ".join(ph_list)

            # collect words overlapping this span
            w_text = []
            for w in words:
                wb, we = float(w["begin"]), float(w["end"])
                if we <= t0: 
                    continue
                if wb >= t1:
                    break
                w_text.append(w["word"])
            text = " ".join(w_text).strip()

            # filter meaningless text
            if args.drop_empty and not text:
                dropped += 1
                continue
            if not is_meaningful_text(text, args.min_chars, bool(args.drop_nonhangul), bool(args.keep_punct_only)):
                dropped += 1
                continue

            text_ref = surface_from_text(text) if text else ""
            jamos = hangul_to_jamo(text) if text else ""
            jamos_ref = hangul_to_jamo(text_ref) if text_ref else ""

            # surface phones via IPA phonology (if available)
            ph_ref_list, rule_tags = surface_from_ipa_list(ph_list)
            phonemes = ph_str
            phonemes_ref = " ".join(ph_ref_list)

            clip_path = ""
            if args.write_wavs:
                clip_path = (out_wavs / f"{key}_{kept:06d}.wav")
                try:
                    ffmpeg_cut(audio, clip_path, t0, t1, args.pad_pre_ms, args.pad_post_ms)
                    clip_path = str(clip_path)
                except subprocess.CalledProcessError:
                    clip_path = ""

            item = {
                "index": kept,
                "file": key,
                "speaker": ph_chunk[0].get("speaker",""),
                "t_begin": t0,
                "t_end": t1,
                "duration": round(t1 - t0, 3),
                "audio": clip_path,
                "text": text,
                "text_ref": text_ref,
                "jamos": jamos,
                "jamos_ref": jamos_ref,
                "phonemes": phonemes,
                "phonemes_ref": phonemes_ref,
                "n_phones": len(ph_list),
                "phones": ph_list,
                "rule_tags": rule_tags,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Kept {kept} segments, dropped {dropped}.")
    print(f"Manifest: {out_manifest}")
    if args.write_wavs:
        print(f"WAVs saved under {out_wavs}")
