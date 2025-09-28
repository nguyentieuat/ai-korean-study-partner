#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_oov_lexicon.py
Aggregate OOV words from ASR outputs, generate pronunciation candidates (text_ref/g2pk2/jamo),
and produce an extended MFA dictionary.

Inputs (choose one of the ASR sources):
  --from_jsonl slices_asr.jsonl        # lines with {"audio":..., "text": "...", "text_ref": "...", "jamos_ref": "...", "phonemes_ref": ...}
  --from_csv   manifest_with_asr.csv   # CSV with columns: asr_text, text_ref, jamos_ref, phonemes_ref (script will try best-effort)

Other required input:
  --base_dict  korean_mfa.dict         # your existing MFA lexicon (WORD PH1 PH2 ...)

Outputs:
  - oov_candidates.csv     # review table
  - oov_candidates.dict    # MFA-style proposed entries (WORD PH1 PH2 ...)
  - korean_mfa.extended.dict   # merged base_dict + oov (deduped)
  - (optional) debug JSONL with extra fields

Phoneset options:
  --phoneset jamo          # use NFD jamo as phones (only if your AM uses jamo phones!)
  --phoneset g2pk2         # use g2pk2 on surface (text_ref) as phones
  --phoneset map_json      # supply --jamo_map_json to map Jamo -> your MFA phoneset

Usage example:
  python build_oov_lexicon.py \
    --from_jsonl "E:/.../output_vad/slices_asr.jsonl" \
    --base_dict "E:/.../korean_mfa.dict" \
    --phoneset g2pk2 --min_freq 2 --min_len 2 --out_dir "E:/.../lex_out"

Notes:
- This script is conservative: filters low-frequency, non-Hangul tokens, and keeps up to two variants per word
  (plain text and surface text_ref if different).
- You can later open oov_candidates.csv to review before using the extended dict for MFA.
"""

import os, re, csv, json, unicodedata as ud
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Iterable
from collections import Counter, defaultdict

# Optional: ko_phonology for surface form
try:
    from ko_phonology import explain_clean_for_user  # (surface_text, rule_tags)
except Exception:
    explain_clean_for_user = None

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".wma", ".aac"}
HANGUL_ONLY = re.compile(r'[\uAC00-\uD7A3]+')
JAMO_BLOCK = re.compile(r'[\u1100-\u11FF\u3130-\u318F]')
_NUM_RE = re.compile(r"^[0-9]+(\.[0-9]+)?$")
_BAD_TOKENS = {"SIL", "SP", "SILENCE", "NSN", "SPN"}

def norm_nfc(s: str) -> str:
    return ud.normalize("NFC", (s or "").strip())

def nfd_jamos(s: str) -> List[str]:
    nfd = ud.normalize("NFD", s or "")
    return [ch for ch in nfd if JAMO_BLOCK.match(ch)]

def tokenize_hangul(text: str) -> List[str]:
    toks = []
    for tok in (text or "").split():
        toks.extend(HANGUL_ONLY.findall(tok))
    return toks

def sanitize_pron_tokens(tokens: Iterable[str]) -> List[str]:
    clean = []
    for t in tokens:
        if not t: continue
        if t in _BAD_TOKENS: continue
        if _NUM_RE.match(t): continue
        clean.append(t)
    return clean

def load_mfa_dict(path: Path) -> Dict[str, List[List[str]]]:
    lex = {}
    if not path.exists():
        return lex
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith(";"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            word = parts[0]
            pron = parts[1:]
            pron = sanitize_pron_tokens(pron)
            if not pron:
                continue
            lex.setdefault(word, []).append(pron)
    return lex

def dump_mfa_dict(path: Path, lex: Dict[str, List[List[str]]]):
    with path.open("w", encoding="utf-8") as f:
        for w, prons in sorted(lex.items()):
            for p in prons:
                f.write(w + " " + " ".join(p) + "\n")

def try_g2pk2(text_ref: str) -> List[str]:
    try:
        from g2pk2 import G2p
        _G = G2p()
        out = _G(text_ref)
        if isinstance(out, str): 
            return sanitize_pron_tokens(list(out))
        return sanitize_pron_tokens(out)
    except Exception:
        return []

def maybe_surface(text: str) -> str:
    if explain_clean_for_user is None:
        return text
    try:
        surface, _ = explain_clean_for_user(text)
        return norm_nfc(surface)
    except Exception:
        return text

def load_asr_jsonl(path: Path) -> List[str]:
    words = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                o = json.loads(line)
            except Exception:
                continue
            text = norm_nfc(o.get("text") or "")
            if not text: 
                continue
            words.extend(tokenize_hangul(text))
    return words

def load_asr_csv(path: Path) -> List[str]:
    words = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            txt = norm_nfc(row.get("asr_text") or row.get("text") or "")
            if not txt:
                continue
            words.extend(tokenize_hangul(txt))
    return words

def map_jamo_with_table(jamos: List[str], mapping: Dict[str,str]) -> List[str]:
    out = []
    for j in jamos:
        ph = mapping.get(j)
        if ph: out.append(ph)
    return sanitize_pron_tokens(out)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Aggregate OOV words and build extended MFA lexicon")
    ap.add_argument("--from_jsonl", type=str, default=None, help="slices_asr.jsonl")
    ap.add_argument("--from_csv", type=str, default=None, help="manifest_with_asr.csv")
    ap.add_argument("--base_dict", type=str, required=True, help="Existing MFA dict")
    ap.add_argument("--phoneset", type=str, default="g2pk2", choices=["g2pk2","jamo","map_json"],
                    help="How to generate phones for OOV")
    ap.add_argument("--jamo_map_json", type=str, default=None, help="JSON mapping Jamo char -> phone symbol (for phoneset=map_json)")
    ap.add_argument("--min_freq", type=int, default=2, help="Only include OOV with frequency >= this")
    ap.add_argument("--min_len", type=int, default=2, help="Minimum Hangul token length to consider")
    ap.add_argument("--max_variants_per_word", type=int, default=2, help="Keep up to N variants per word")
    ap.add_argument("--out_dir", type=str, required=True, help="Output folder")
    args = ap.parse_args()

    OUT = Path(args.out_dir); OUT.mkdir(parents=True, exist_ok=True)
    base_dict = load_mfa_dict(Path(args.base_dict))

    # 1) Collect words from ASR
    corpus_words = []
    if args.from_jsonl:
        corpus_words += load_asr_jsonl(Path(args.from_jsonl))
    if args.from_csv:
        corpus_words += load_asr_csv(Path(args.from_csv))
    if not corpus_words:
        print("No ASR text found. Provide --from_jsonl or --from_csv."); raise SystemExit

    # 2) Count and filter
    from collections import Counter
    freq = Counter([w for w in corpus_words if len(w) >= args.min_len])
    # 3) OOV vs base dict keys
    oov = [w for w,c in freq.items() if w not in base_dict and c >= args.min_freq]

    print(f"Total unique tokens: {len(freq)} | OOV above freq≥{args.min_freq}: {len(oov)}")

    # Load mapping if requested
    jamo_map = None
    if args.phoneset == "map_json":
        if not args.jamo_map_json: 
            print("phoneset=map_json requires --jamo_map_json"); raise SystemExit
        with open(args.jamo_map_json, "r", encoding="utf-8") as f:
            jamo_map = json.load(f)

    # 4) Propose pronunciations
    proposals = {}  # word -> list of pron variants (list[str])
    rows = []
    for w in sorted(oov):
        text = norm_nfc(w)
        # surface form
        text_ref = maybe_surface(text)

        variants = []
        # choose phones
        if args.phoneset == "g2pk2":
            ph = try_g2pk2(text_ref)
            if not ph:
                ph = nfd_jamos(text_ref)  # fallback
            if ph:
                variants.append(ph)
        elif args.phoneset == "jamo":
            ph = nfd_jamos(text_ref)
            if ph:
                variants.append(ph)
        elif args.phoneset == "map_json":
            jamos = nfd_jamos(text_ref)
            ph = map_jamo_with_table(jamos, jamo_map or {})
            if ph:
                variants.append(ph)

        # Optionally include a second variant using *raw text* if surface differs
        if text_ref != text:
            if args.phoneset == "g2pk2":
                ph2 = try_g2pk2(text)
                if ph2:
                    variants.append(ph2)
            elif args.phoneset == "jamo":
                variants.append(nfd_jamos(text))
            elif args.phoneset == "map_json":
                variants.append(map_jamo_with_table(nfd_jamos(text), jamo_map or {}))

        # Dedup, truncate
        uniq = []
        seen = set()
        for v in variants:
            key = " ".join(v)
            if key not in seen and v:
                uniq.append(v); seen.add(key)
            if len(uniq) >= args.max_variants_per_word:
                break

        if uniq:
            proposals[w] = uniq
            for i, phs in enumerate(uniq, 1):
                rows.append([w, i, text_ref, " ".join(phs), freq[w]])
        else:
            rows.append([w, 0, text_ref, "", freq[w]])

    # 5) Write review CSV and dicts
    csv_path = OUT / "oov_candidates.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["word","variant_idx","surface_text_ref","phones","freq"])
        wcsv.writerows(rows)

    # Proposed OOV dict
    oov_dict_path = OUT / "oov_candidates.dict"
    with oov_dict_path.open("w", encoding="utf-8") as f:
        for w, prons in sorted(proposals.items()):
            for p in prons:
                f.write(w + " " + " ".join(p) + "\n")

    # Merge with base dict
    merged = dict(base_dict)
    for w, prons in proposals.items():
        existing = merged.get(w, [])
        for p in prons:
            if p not in existing:
                existing.append(p)
        merged[w] = existing
    merged_path = OUT / "korean_mfa.extended.dict"
    dump_mfa_dict(merged_path, merged)

    # Debug JSONL
    dbg_path = OUT / "oov_debug.jsonl"
    with dbg_path.open("w", encoding="utf-8") as f:
        for w, prons in proposals.items():
            rec = {"word": w, "freq": freq[w], "text_ref": maybe_surface(w), "phones": prons}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("Wrote:")
    print(" -", csv_path.resolve())
    print(" -", oov_dict_path.resolve())
    print(" -", merged_path.resolve())
    print(" -", dbg_path.resolve())
