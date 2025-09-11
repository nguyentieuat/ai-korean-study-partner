# %%writefile text_to_ipa_with_mfa.py
# filename: text_to_ipa_with_mfa.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert: audio_path | sid | text  -->  audio_path | sid | IPA

Ưu tiên: korean_mfa.dict (theo TỪ).
- Dòng: <word> <một-vài số> <chuỗi IPA tách bằng space>
- Nếu một từ có nhiều dòng, chọn dòng có score đầu tiên cao nhất.

Fallback OOV:
- (tùy) g2pk2 để chuẩn hoá (có thể tắt bằng --no-g2pk2).
- Áp dụng quy tắc biến âm trong PHẠM VI TỪ để cải thiện phonemize.
- phonemize qua phonemizer/espeak-ng (CLI fallback), rồi (tùy) compact + bỏ diacritics.

Có tuỳ chọn: chỉ chạy train, compact IPA theo từ, bỏ diacritics.
"""

import io, os, sys, re, shlex, argparse, subprocess
from typing import Dict, List, Tuple
from tqdm import tqdm

# -------- Optional deps --------
try:
    from g2pk2 import G2p
    HAS_G2PK2 = True
except Exception:
    G2PK2_ERR = sys.exc_info()
    HAS_G2PK2 = False

try:
    from phonemizer import phonemize
    HAS_PHONEMIZER = True
except Exception:
    PHONEMIZER_ERR = sys.exc_info()
    HAS_PHONEMIZER = False


# ----------------- I/O helpers -----------------
def safe_split_line(line: str) -> Tuple[str, str, str]:
    parts = [p.strip() for p in line.rstrip("\n").split("|", 2)]
    if len(parts) != 3:
        raise ValueError(f"Line must have 3 fields separated by '|': {line[:200]}")
    return parts[0], parts[1], parts[2]

def read_triples(path: str) -> List[Tuple[str, str, str]]:
    rows = []
    with io.open(path, "r", encoding="utf-8") as f:
        for raw in f:
            if not raw.strip():
                continue
            try:
                rows.append(safe_split_line(raw))
            except Exception as e:
                sys.stderr.write(f"[WARN] skip malformed line: {e}\n")
    return rows

def write_triples(path: str, triples: List[Tuple[str, str, str]], ipa_list: List[str]):
    assert len(triples) == len(ipa_list)
    with io.open(path, "w", encoding="utf-8") as f:
        for (a, sid, _), ipa in zip(triples, ipa_list):
            f.write(f"{a}|{sid}|{ipa}\n")


# ----------------- Dict loading -----------------
_FLOAT_RE = re.compile(r'^[+-]?(?:\d+\.?\d*|\.\d+)$')
def _is_float(tok: str) -> bool:
    return bool(_FLOAT_RE.match(tok))

def load_mfa_dict(dict_path: str) -> Dict[str, str]:
    """
    Parse korean_mfa.dict:
        <word> <nhiều số?> <IPA ...>
    Trả về: dict[word] -> "k ɐ ɟ ʌ k̚"
    Nếu nhiều dòng cùng 1 từ -> chọn dòng có score đầu tiên lớn nhất.
    """
    best: Dict[str, Tuple[float, str]] = {}
    n_lines = 0
    with io.open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = re.split(r"\s+", line)
            if not cols: 
                continue
            word = cols[0]
            j = 1
            scores = []
            while j < len(cols) and _is_float(cols[j]):
                scores.append(float(cols[j])); j += 1
            if j >= len(cols):  # không có IPA
                continue
            pron = " ".join(cols[j:])
            prob = scores[0] if scores else 1.0
            prev = best.get(word)
            if (prev is None) or (prob > prev[0]):
                best[word] = (prob, pron)
    lex = {w: p for w, (sc, p) in best.items()}
    print(f"[INFO] loaded dict lines: {n_lines:,} | entries: {len(lex):,}")
    return lex


# ----------------- Tokenization & cleanup -----------------
PUNCT = "“”\"'‘’.·…,:;!?()[]{}<>/|~`—–-"
TRIM_RE = re.compile(f"^[{re.escape(PUNCT)}]+|[{re.escape(PUNCT)}]+$")

def clean_token_for_lookup(tok: str) -> str:
    return TRIM_RE.sub("", tok)


# ----------------- espeak helpers -----------------
def _ensure_espeak_library_env():
    if os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
        return
    for p in [
        "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",
        "/usr/lib/aarch64-linux-gnu/libespeak-ng.so.1",
        "/usr/local/lib/libespeak-ng.so.1",
        "/usr/lib/libespeak-ng.so.1",
    ]:
        if os.path.exists(p):
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = p
            return

def _espeak_cli_single(text: str) -> str:
    cmd = os.environ.get("ESPEAK_NG_CMD", "espeak-ng --ipa -v ko -q --stdout")
    out = subprocess.run(
        shlex.split(cmd), input=text, text=True,
        capture_output=True, check=True, timeout=20
    ).stdout
    return " ".join(out.split())

def phonemize_ko(texts: List[str], strip: bool, preserve_punctuation: bool, jobs: int) -> List[str]:
    if HAS_PHONEMIZER:
        try:
            _ensure_espeak_library_env()
            res = phonemize(
                texts, language="ko", backend="espeak",
                strip=strip, preserve_punctuation=preserve_punctuation,
                njobs=jobs, with_stress=False,
                punctuation_marks=';:,.!?¡¿—…"«»“”()[]{}'
            )
            if isinstance(res, list):
                return [" ".join(r.split()) for r in res]
            return [" ".join(res.split())]
        except Exception as e:
            sys.stderr.write(f"[WARN] phonemizer/espeak failed ({e}); fallback to espeak-ng CLI.\n")
    return [_espeak_cli_single(t) for t in texts]

def ensure_fallback_available():
    if not HAS_PHONEMIZER:
        raise RuntimeError(
            "phonemizer missing (pip install phonemizer) và cần espeak-ng (apt-get -y install espeak-ng)"
        )


# ----------------- Hangul jamo decomposition -----------------
L_CHO = list("ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄋᄌᄍᄎᄏᄐᄑᄒ")
V_JUNG = list("ᅡᅢᅣᅤᅥᅦᅧᅨᅩᅪᅫᅬᅭᅮᅯᅰᅱᅲᅳᅴᅵ")
T_JONG = ["" ,"ᆨ","ᆩ","ᆪ","ᆫ","ᆬ","ᆭ","ᆮ","ᆯ","ᆰ","ᆱ","ᆲ","ᆳ","ᆴ","ᆵ","ᆶ","ᆷ","ᆸ","ᆹ","ᆺ","ᆻ","ᆼ","ᆽ","ᆾ","ᆿ","ᇀ","ᇁ","ᇂ"]
L2I = {c:i for i,c in enumerate(L_CHO)}
V2I = {c:i for i,c in enumerate(V_JUNG)}
T2I = {c:i for i,c in enumerate(T_JONG)}

H_BASE = 0xAC00
N_L, N_V, N_T = 19, 21, 28
CHO_ㅇ = L2I["ᄋ"]; JONG_NONE = 0

def is_hangul_syll(ch: str) -> bool:
    return 0xAC00 <= ord(ch) <= 0xD7A3

def decompose(ch: str) -> Tuple[int,int,int]:
    code = ord(ch)
    if not (0xAC00 <= code <= 0xD7A3):
        return -1,-1,-1
    s = code - H_BASE
    L = s // (N_V * N_T)
    V = (s % (N_V * N_T)) // N_T
    T = s % N_T
    return L,V,T

def compose(L:int,V:int,T:int) -> str:
    return chr(H_BASE + (L * N_V + V) * N_T + T)

# Utility groups
JONG_K = {"ᆨ","ᆩ","ᆪ","ᆿ"}
JONG_T = {"ᆮ","ᆺ","ᆻ","ᆽ","ᆾ","ᇀ","ᇂ"}  # t-class
JONG_P = {"ᆸ","ᆹ","ᇁ"}
JONG_N = {"ᆫ"}; JONG_M = {"ᆷ"}; JONG_L = {"ᆯ"}; JONG_NG = {"ᆼ"}; JONG_H={"ᇂ"}

LENIS = {"ᄀ","ᄃ","ᄇ","ᄉ","ᄌ"}
TENSE = {"ᄀ":"ᄁ","ᄃ":"ᄄ","ᄇ":"ᄈ","ᄉ":"ᄊ","ᄌ":"ᄍ"}
ASP_MAP = {"ᄀ":"ᄏ","ᄃ":"ᄐ","ᄇ":"ᄑ","ᄌ":"ᄎ"}

V_FRONT_I = {"ᅵ"}           # cho 구개음화 tối thiểu
V_Y_VOWELS = {"ᅵ","ᅣ","ᅧ","ᅭ","ᅲ"}  # cho ㄴ첨가

def apply_rules_in_word(word: str) -> str:
    """Áp dụng các quy tắc biến âm trong PHẠM VI 1 TỪ (không vượt qua khoảng trắng)."""
    if not word:
        return word
    syls = []
    for ch in word:
        L,V,T = decompose(ch)
        if L<0: syls.append(ch)
        else:   syls.append([L,V,T])

    for i in range(len(syls)-1):
        if not (isinstance(syls[i], list) and isinstance(syls[i+1], list)):
            continue
        L1,V1,T1 = syls[i]
        L2,V2,T2 = syls[i+1]

        jong1 = T_JONG[T1]
        cho2  = L_CHO[L2]
        jung2 = V_JUNG[V2]

        # 6) ㅎ 탈락
        if jong1 in JONG_H and L2 == CHO_ㅇ:
            T1 = JONG_NONE
            jong1 = ""

        # 7) 격음화
        if (jong1 in JONG_K or jong1 in JONG_T) and cho2 == "ᄒ":
            if jong1 in JONG_K: cho2 = "ᄏ"; L2 = L2I[cho2]; T1 = JONG_NONE; jong1=""
            elif jong1 in JONG_T: cho2 = "ᄐ"; L2 = L2I[cho2]; T1 = JONG_NONE; jong1=""
        if jong1 in JONG_H and cho2 in ("ᄀ","ᄃ"):
            cho2 = ASP_MAP[cho2]; L2 = L2I[cho2]; T1 = JONG_NONE; jong1=""

        # 3) 비음화
        if (jong1 in JONG_K and cho2 in ("ᄂ","ᄆ")):
            T1 = T2I["ᆼ"]; jong1="ᆼ"
        elif (jong1 in JONG_T and cho2 in ("ᄂ","ᄆ")):
            T1 = T2I["ᆫ"]; jong1="ᆫ"
        elif (jong1 in JONG_P and cho2 in ("ᄂ","ᄆ")):
            T1 = T2I["ᆷ"]; jong1="ᆷ"
        if (jong1 in JONG_M or jong1 in JONG_NG) and cho2=="ᄅ":
            L2 = L2I["ᄂ"]; cho2="ᄂ"
        if (jong1 in JONG_K or jong1 in JONG_P) and cho2=="ᄅ":
            L2 = L2I["ᄂ"]; cho2="ᄂ"

        # 4) 유음화
        if jong1 in JONG_L and cho2=="ᄂ":
            L2 = L2I["ᄅ"]; cho2="ᄅ"
        if jong1 in JONG_N and cho2=="ᄅ":
            T1 = T2I["ᆯ"]; jong1="ᆯ"

        # 5) 구개음화
        if (jong1 in {"ᆮ","ᇀ"}) and L2==CHO_ㅇ and jung2 in V_FRONT_I:
            L2 = L2I["ᄌ"] if jong1=="ᆮ" else L2I["ᄎ"]
            T1 = JONG_NONE
            cho2 = L_CHO[L2]; jong1=""

        # 2) 경음화
        if cho2 in LENIS:
            trigger = False
            if jong1 in {"ᆨ","ᆮ","ᆸ"}: trigger = True
            if jong1 in {"ᆫ","ᆷ","ᆯ"} and cho2 in {"ᄀ","ᄃ","ᄌ"}: trigger = True
            if jong1 in JONG_L and cho2 in {"ᄀ","ᄃ","ᄉ","ᄌ"}: trigger = True
            if trigger:
                cho2 = TENSE.get(cho2, cho2); L2 = L2I[cho2]

        # 1) 연음화
        if L2 == CHO_ㅇ and jong1:
            move_map = {"ᆨ":"ᄀ","ᆮ":"ᄃ","ᆸ":"ᄇ","ᆫ":"ᄂ","ᆷ":"ᄆ","ᆯ":"ᄅ","ᆼ":"ᄋ"}
            if jong1 in move_map:
                L2 = L2I[move_map[jong1]]; cho2 = L_CHO[L2]
                T1 = JONG_NONE; jong1=""

        # 9) ㄴ첨가 (đơn giản)
        if L2 == CHO_ㅇ and jung2 in V_Y_VOWELS and T1 != JONG_NONE:
            L2 = L2I["ᄂ"]; cho2="ᄂ"

        syls[i]   = [L1,V1,T1]
        syls[i+1] = [L2,V2,T2]

    out = []
    for s in syls:
        if isinstance(s, list): out.append(compose(s[0],s[1],s[2]))
        else: out.append(s)
    return "".join(out)


# ----------------- IPA post-processing -----------------
IPA_REMOVE = "".join([
    "\u02D0",  # ː
    "\u02B0",  # ʰ
    "\u02B2",  # ʲ
    "\u02B7",  # ʷ
    "\u0361",  # ͡
    "\u031A",  # ̚
    "\u0348",  # ͈
])
IPA_REMOVE_RE = re.compile("[" + re.escape(IPA_REMOVE) + "]")

def strip_ipa_diacritics(s: str) -> str:
    return IPA_REMOVE_RE.sub("", s)

def compact_ipa_word(s: str) -> str:
    return s.replace(" ", "")


# ----------------- g2pk2 safe normalize -----------------
_SENT_SPLIT_RE = re.compile(r'([\.!?…]+|\n+)')

def safe_normalize(text: str, g2p_obj, max_chunk: int = 120) -> str:
    """Chuẩn hoá an toàn: chia nhỏ câu, bắt MemoryError/Exception và fallback."""
    if g2p_obj is None:
        return text
    try:
        if len(text) <= max_chunk:
            return g2p_obj(text)
        parts = _SENT_SPLIT_RE.split(text)
        out = []
        buf = ""
        for p in parts:
            if not p:
                continue
            if _SENT_SPLIT_RE.fullmatch(p):
                # flush buffer
                if buf:
                    out.append(g2p_obj(buf))
                    buf = ""
                out.append(p)
            else:
                # p là đoạn thường; nếu quá dài thì tách theo khoảng trắng
                if len(p) > max_chunk:
                    toks = p.split()
                    cur = []
                    cur_len = 0
                    for t in toks:
                        if cur_len + len(t) + 1 > max_chunk:
                            out.append(g2p_obj(" ".join(cur)))
                            cur = [t]; cur_len = len(t)
                        else:
                            cur.append(t); cur_len += len(t) + 1
                    if cur:
                        out.append(g2p_obj(" ".join(cur)))
                else:
                    buf += p
        if buf:
            out.append(g2p_obj(buf))
        return "".join(out)
    except MemoryError:
        sys.stderr.write("[WARN] g2pk2 MemoryError -> fallback to raw text for this line.\n")
        return text
    except Exception as e:
        sys.stderr.write(f"[WARN] g2pk2 failed ({e}) -> fallback to raw text for this line.\n")
        return text


# ----------------- Fallback (sentence) -----------------
def fallback_sent_to_ipa(sent: str, g2p_obj, jobs=2, preserve_punct=False) -> str:
    norm = safe_normalize(sent, g2p_obj)
    ipa = phonemize_ko([norm], strip=not preserve_punct,
                       preserve_punctuation=preserve_punct, jobs=jobs)[0]
    return ipa


# ----------------- Core processing -----------------
def sentence_to_ipa_by_dict(
    sent: str,
    lex: Dict[str, str],
    no_fallback: bool,
    g2p_obj=None,
    jobs: int = 2,
    per_token_fallback: bool = True,
    apply_rules: bool = True,
    compact_words: bool = True,
    strip_diacritics: bool = True,
) -> str:
    norm = safe_normalize(sent, g2p_obj)
    raw_tokens = norm.split()

    ipa_tokens: List[str] = []

    for raw_tok in raw_tokens:
        key = clean_token_for_lookup(raw_tok)
        if not key:
            continue

        pron = lex.get(key)
        if pron is None:
            if no_fallback:
                ipa = f"<OOV:{key}>"
            else:
                tok_for_g2p = apply_rules_in_word(key) if apply_rules else key
                ipa = phonemize_ko([tok_for_g2p], strip=True, preserve_punctuation=False, jobs=jobs)[0]
        else:
            ipa = pron

        if strip_diacritics:
            ipa = strip_ipa_diacritics(ipa)
        if compact_words:
            ipa = compact_ipa_word(ipa)

        ipa_tokens.append(ipa)

    return " ".join(ipa_tokens)


def process_file(inp: str, outp: str, lex: Dict[str, str],
                 no_fallback: bool, jobs: int,
                 sentence_fallback: bool,
                 apply_rules: bool,
                 compact_words: bool,
                 strip_diacritics: bool,
                 disable_g2pk2: bool):
    rows = read_triples(inp)
    if not rows:
        print(f"[WARN] empty or invalid: {inp}")
        return

    # g2p object (có thể bị tắt)
    g2p_obj = None
    if (not disable_g2pk2) and HAS_G2PK2:
        try:
            g2p_obj = G2p()
        except Exception as e:
            sys.stderr.write(f"[WARN] G2p init failed ({e}) -> run without g2pk2.\n")
            g2p_obj = None

    if (not no_fallback):
        ensure_fallback_available()

    out_list: List[str] = []
    for _, _, txt in tqdm(rows, desc=f"to IPA ({os.path.basename(inp)})"):
        try:
            per_token = (not sentence_fallback)
            if (not no_fallback) and (not per_token):
                ipa = fallback_sent_to_ipa(txt, g2p_obj, jobs=jobs, preserve_punct=False)
                if strip_diacritics:
                    ipa = strip_ipa_diacritics(ipa)
            else:
                ipa = sentence_to_ipa_by_dict(
                    txt, lex, no_fallback=no_fallback, g2p_obj=g2p_obj, jobs=jobs,
                    per_token_fallback=True, apply_rules=apply_rules,
                    compact_words=compact_words, strip_diacritics=strip_diacritics
                )
        except MemoryError:
            sys.stderr.write("[WARN] MemoryError on line -> fallback to phonemize(raw).\n")
            ipa = phonemize_ko([txt], strip=True, preserve_punctuation=False, jobs=jobs)[0]
            if strip_diacritics: ipa = strip_ipa_diacritics(ipa)
        except Exception as e:
            sys.stderr.write(f"[WARN] line failed ({e}) -> fallback to phonemize(raw).\n")
            ipa = phonemize_ko([txt], strip=True, preserve_punctuation=False, jobs=jobs)[0]
            if strip_diacritics: ipa = strip_ipa_diacritics(ipa)

        out_list.append(ipa)

    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    write_triples(outp, rows, out_list)
    print(f"[OK] wrote {outp}  (lines={len(rows)})")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict", required=True, help="path to korean_mfa.dict")
    ap.add_argument("--train", required=True, help="train: audio|sid|text")
    ap.add_argument("--val",   required=True, help="val:   audio|sid|text")
    ap.add_argument("--train-out", default=None, help="default: <train>_ipa.txt")
    ap.add_argument("--val-out",   default=None, help="default: <val>_ipa.txt")
    ap.add_argument("--jobs", type=int, default=2, help="CPU workers for phonemizer")
    ap.add_argument("--no-fallback", action="store_true",
                    help="Only use dict; OOV -> <OOV:tok>. If omitted, use phonemizer fallback.")
    ap.add_argument("--fallback-sentence", action="store_true",
                    help="When falling back, phonemize the WHOLE sentence instead of per-token.")
    ap.add_argument("--espeak-lib", default="",
                    help="Optional path to libespeak-ng.so.1 -> PHONEMIZER_ESPEAK_LIBRARY.")
    ap.add_argument("--only-train", action="store_true", help="Chỉ xử lý file train.")
    ap.add_argument("--no-apply-rules", action="store_true", help="Tắt quy tắc biến âm trong phạm vi từ (OOV).")
    ap.add_argument("--no-compact-words", action="store_true", help="Không gộp IPA trong từ.")
    ap.add_argument("--no-strip-diacritics", action="store_true", help="Giữ nguyên diacritics IPA.")
    ap.add_argument("--no-g2pk2", action="store_true", help="Tắt g2pk2 (tránh lỗi MeCab/thiếu RAM).")
    return ap.parse_args()


def main():
    args = parse_args()

    if args.espeak_lib:
        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = args.espeak_lib
    else:
        _ensure_espeak_library_env()

    lex = load_mfa_dict(args.dict)
    print(f"[INFO] dict entries ready: {len(lex):,}")

    train_out = args.train_out or (os.path.splitext(args.train)[0] + "_ipa.txt")
    val_out   = args.val_out   or (os.path.splitext(args.val)[0]   + "_ipa.txt")

    process_file(
        args.train, train_out, lex,
        no_fallback=args.no_fallback,
        jobs=args.jobs,
        sentence_fallback=args.fallback_sentence,
        apply_rules=not args.no_apply_rules,
        compact_words=not args.no_compact_words,
        strip_diacritics=not args.no_strip_diacritics,
        disable_g2pk2=args.no_g2pk2
    )
    if not args.only_train:
        process_file(
            args.val, val_out, lex,
            no_fallback=args.no_fallback,
            jobs=args.jobs,
            sentence_fallback=args.fallback_sentence,
            apply_rules=not args.no_apply_rules,
            compact_words=not args.no_compact_words,
            strip_diacritics=not args.no_strip_diacritics,
            disable_g2pk2=args.no_g2pk2
        )

if __name__ == "__main__":
    main()
