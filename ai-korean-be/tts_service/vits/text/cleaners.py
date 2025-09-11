""" from https://github.com/keithito/tacotron """

"""
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
"""

import re
from unidecode import unidecode
from phonemizer import phonemize


# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
  ('mrs', 'misess'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort'),
]]


def expand_abbreviations(text):
  for regex, replacement in _abbreviations:
    text = re.sub(regex, replacement, text)
  return text


def expand_numbers(text):
  return normalize_numbers(text)


def lowercase(text):
  return text.lower()


def collapse_whitespace(text):
  return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
  return unidecode(text)


def basic_cleaners(text):
  """Basic pipeline that lowercases and collapses whitespace without transliteration."""
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text


def transliteration_cleaners(text):
  """Pipeline for non-English text that transliterates to ASCII."""
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text


def english_cleaners(text):
  """Pipeline for English text, including abbreviation expansion."""
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = expand_abbreviations(text)
  phonemes = phonemize(text, language='en-us', backend='espeak', strip=True)
  phonemes = collapse_whitespace(phonemes)
  return phonemes


def english_cleaners2(text):
  """Pipeline for English text, including abbreviation expansion. + punctuation + stress"""
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = expand_abbreviations(text)
  phonemes = phonemize(text, language='en-us', backend='espeak', strip=True, preserve_punctuation=True, with_stress=True)
  phonemes = collapse_whitespace(phonemes)
  return phonemes


"""
Korean cleaners (Jamo pipeline) for VITS/TTS.

- Numbers → Korean words:
  * Contextual: native (thuần Hàn) cho 단위 như 시/시간/살/명..., sino (Hán–Hàn) cho 월/년/분...
  * Đặc biệt: "N시 반", 6월→유 월, 10월→시 월.
- Hangul → Compatibility Jamo:
  * Mỗi jamo cách nhau 1 khoảng trắng (jamo_sep=' ')
  * Giữa các âm tiết Hangul chèn dấu '·' (syl_sep='·') — KHÔNG chèn qua biên từ hoặc trước/ sau punctuation.
  * Dấu cách giữa từ chuyển thành space_token (mặc định '▁') trước khi phân rã.

Ví dụ:
  "오늘 3시 meeting 있어요."
  → "ㅇ ㅗ ㄴ·ㅡ ㄹ ▁ ㅅ ㅔ ▁ ㅅ ㅣ ▁ m e e t i n g ▁ ㅇ ㅣ ㅆ·ㅇ ㅓ·ㅇ ㅛ ."
"""

from __future__ import annotations
import re
import unicodedata
from typing import Optional, List, Tuple

# ---------- whitespace ----------
_WS_RE = re.compile(r"\s+")
def collapse_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()

# ---------- Hangul Jamo tables (compat) ----------
# U+1100.. : choseong/jungseong/jongseong → map sang compatibility Jamo U+3131..
CHOSUNG_TO_COMPAT = {
    0x1100:"ㄱ",0x1101:"ㄲ",0x1102:"ㄴ",0x1103:"ㄷ",0x1104:"ㄸ",0x1105:"ㄹ",
    0x1106:"ㅁ",0x1107:"ㅂ",0x1108:"ㅃ",0x1109:"ㅅ",0x110A:"ㅆ",0x110B:"ㅇ",
    0x110C:"ㅈ",0x110D:"ㅉ",0x110E:"ㅊ",0x110F:"ㅋ",0x1110:"ㅌ",0x1111:"ㅍ",0x1112:"ㅎ",
}
JUNGSUNG_TO_COMPAT = {
    0x1161:"ㅏ",0x1162:"ㅐ",0x1163:"ㅑ",0x1164:"ㅒ",0x1165:"ㅓ",0x1166:"ㅔ",
    0x1167:"ㅕ",0x1168:"ㅖ",0x1169:"ㅗ",0x116A:"ㅘ",0x116B:"ㅙ",0x116C:"ㅚ",
    0x116D:"ㅛ",0x116E:"ㅜ",0x116F:"ㅝ",0x1170:"ㅞ",0x1171:"ㅟ",0x1172:"ㅠ",
    0x1173:"ㅡ",0x1174:"ㅢ",0x1175:"ㅣ",
}
JONGSUNG_TO_COMPAT = {
    0x11A8:"ㄱ",0x11A9:"ㄲ",0x11AA:"ㄳ",0x11AB:"ㄴ",0x11AC:"ㄵ",0x11AD:"ㄶ",
    0x11AE:"ㄷ",0x11AF:"ㄹ",0x11B0:"ㄺ",0x11B1:"ㄻ",0x11B2:"ㄼ",0x11B3:"ㄽ",
    0x11B4:"ㄾ",0x11B5:"ㄿ",0x11B6:"ㅀ",0x11B7:"ㅁ",0x11B8:"ㅂ",0x11B9:"ㅄ",
    0x11BA:"ㅅ",0x11BB:"ㅆ",0x11BC:"ㅇ",0x11BD:"ㅈ",0x11BE:"ㅊ",0x11BF:"ㅋ",
    0x11C0:"ㅌ",0x11C1:"ㅍ",0x11C2:"ㅎ",
}

# ---------- Numeral dictionaries ----------
_SINO_DIGITS = ["영","일","이","삼","사","오","육","칠","팔","구"]
_SINO_UNITS4 = ["", "만", "억", "조", "경"]  # 10^0..10^16 (mở rộng được nếu cần)

_NATIVE_1_9 = {1:"한", 2:"두", 3:"세", 4:"네", 5:"다섯", 6:"여섯", 7:"일곱", 8:"여덟", 9:"아홉"}
_NATIVE_TENS = {10:"열", 20:"스물", 30:"서른", 40:"마흔", 50:"쉰",
                60:"예순", 70:"일흔", 80:"여든", 90:"아흔"}

# đơn vị: chọn native hay sino theo ngữ cảnh
UNITS_NATIVE = {
    "시","시간","살","명","마리","개","권","대","병","잔","송이","사람","그루","켤레"
}
UNITS_SINO = {
    "분","초","월","년","일","주","개월","세","층","번","번째","호","호실","원",
    "학년","교시","호선","동","문항","쪽","과","장","페이지"
}

# ---------- Sino-Korean number ----------
def sino_under_10000(n: int) -> str:
    assert 0 <= n < 10000
    parts: List[str] = []
    thou, rem = divmod(n, 1000)
    hund, rem = divmod(rem, 100)
    ten, one  = divmod(rem, 10)

    if thou: parts.append(("" if thou == 1 else _SINO_DIGITS[thou]) + "천")
    if hund: parts.append(("" if hund == 1 else _SINO_DIGITS[hund]) + "백")
    if ten:  parts.append(("" if ten  == 1 else _SINO_DIGITS[ten]) + "십")
    if one:  parts.append(_SINO_DIGITS[one])
    if not parts: parts.append(_SINO_DIGITS[0])
    return "".join(parts)

def sino_number(n: int) -> str:
    if n == 0:
        return _SINO_DIGITS[0]
    chunks: List[str] = []
    unit_idx = 0
    while n > 0:
        n, chunk = divmod(n, 10000)
        if chunk:
            block = sino_under_10000(chunk)
            unit  = _SINO_UNITS4[unit_idx]
            chunks.append(block + unit)
        unit_idx += 1
    return "".join(reversed(chunks))

# ---------- Native Korean number (1..99) ----------
def native_number(n: int, bound: bool = False) -> Optional[str]:
    if n <= 0 or n >= 100:
        return None
    if n in _NATIVE_TENS:
        word = _NATIVE_TENS[n]
        if bound and n == 20:  # only 20 changes 스물→스무 in bound form
            word = "스무"
        return word
    tens = (n // 10) * 10
    ones = n % 10
    pieces: List[str] = []
    if tens:
        t = _NATIVE_TENS[tens]
        # ghép tens+ones: không đổi 스물→스무 (ví dụ 21 → 스물한)
        pieces.append(t)
    if ones:
        pieces.append(_NATIVE_1_9[ones])
    return "".join(pieces) if pieces else None

# ---------- Special phonology for months ----------
def special_month_word(num_hh: str, n: int) -> str:
    # 6월 → 유 월, 10월 → 시 월 (orthography 육/십; TTS thường đọc 유/시)
    if n == 6:  return "유"
    if n == 10: return "시"
    return num_hh

# ---------- Number normalization ----------
_HALF_RE = re.compile(r"(?<!\d)(\d{1,3})\s*시\s*반")
_UNIT_RE = re.compile(
    r"(?<!\d)(\d{1,9})\s*(시|시간|분|초|월|년|일|주|개월|살|세|명|마리|개|권|대|병|잔|송이|층|번|번째|호|호실|원|학년|교시|호선|동)"
)
_PLAIN_NUM_RE = re.compile(r"\b\d{1,12}\b")

def normalize_numbers_ko(text: str) -> str:
    # 1) "N시 반"
    def _half(m: re.Match) -> str:
        n = int(m.group(1))
        native = native_number(n, bound=True) or sino_number(n)
        return f"{native} 시 반"
    text = _HALF_RE.sub(_half, text)

    # 2) "N + UNIT"
    def _unit_repl(m: re.Match) -> str:
        n = int(m.group(1))
        unit = m.group(2)
        use_native = unit in UNITS_NATIVE
        if use_native and 1 <= n < 100:
            num = native_number(n, bound=True)
        else:
            num = sino_number(n)
        if unit == "월" and num in {"육","십"}:
            num = special_month_word(num, n)
        return f"{num} {unit}"
    text = _UNIT_RE.sub(_unit_repl, text)

    # 3) plain numbers → sino
    def _plain(m: re.Match) -> str:
        n = int(m.group(0))
        return sino_number(n)
    text = _PLAIN_NUM_RE.sub(_plain, text)

    return collapse_whitespace(text)

# ---------- Hangul → Jamo with per-syllable separator ----------
def _is_hangul_syllable(ch: str) -> bool:
    code = ord(ch)
    return 0xAC00 <= code <= 0xD7A3

def _decompose_syllable_to_compat_jamo(ch: str, jamo_sep: str) -> str:
    """1 Hangul syllable → 'ㅊ ㅏ ㅇ' (compat jamo with inner spaces)."""
    code = ord(ch)
    SBase, LBase, VBase, TBase = 0xAC00, 0x1100, 0x1161, 0x11A7
    VCount, TCount = 21, 28
    SIndex = code - SBase
    L = LBase + SIndex // (VCount * TCount)
    V = VBase + (SIndex % (VCount * TCount)) // TCount
    T = TBase + SIndex % TCount

    jamos = [CHOSUNG_TO_COMPAT[L], JUNGSUNG_TO_COMPAT[V]]
    if T != TBase:
        jamos.append(JONGSUNG_TO_COMPAT[T])
    return jamo_sep.join(jamos)

def decompose_hangul_segmented(s: str, jamo_sep: str = " ", syl_sep: Optional[str] = "·") -> str:
    """
    Phân rã từng âm tiết Hangul thành jamo (cách nhau bằng jamo_sep).
    Chèn syl_sep giữa HANGUL-ÂM-TIẾT kề nhau; không chèn qua ranh giới từ (▁) hay punctuation.
    """
    tokens: List[Tuple[str, bool]] = []
    for ch in s:
        if _is_hangul_syllable(ch):
            tokens.append((_decompose_syllable_to_compat_jamo(ch, jamo_sep), True))
        else:
            tokens.append((ch, False))

    out: List[str] = []
    for i, (tok, is_han) in enumerate(tokens):
        out.append(tok)
        if syl_sep and is_han:
            # nhìn trước: nếu kế tiếp cũng là Hangul -> chèn syl_sep
            if i + 1 < len(tokens):
                _, next_is_han = tokens[i + 1]
                if next_is_han:
                    out.append(syl_sep)
    return "".join(out)

# ---------- Public cleaner ----------
def korean_cleaners(
    text: str,
    *,
    normalize_numbers: bool = True,
    space_token: Optional[str] = "▁",
    syl_sep: Optional[str] = "·",
    jamo_sep: str = " ",
) -> str:
    """
    Main entry:
      - NFC normalize, collapse spaces
      - (optional) number normalization
      - replace word-space by space_token
      - Hangul → Jamo (with inner jamo spaces) + per-syllable separator
    """
    text = unicodedata.normalize("NFC", text)
    text = collapse_whitespace(text)

    if normalize_numbers:
        text = normalize_numbers_ko(text)
        text = collapse_whitespace(text)

    if space_token:
        text = text.replace(" ", space_token)

    text = decompose_hangul_segmented(text, jamo_sep=jamo_sep, syl_sep=syl_sep)
    text = collapse_whitespace(text)
    return text