# ctc_gate.py — CTC-only gate & confidences for Korean (NFD/Compat/Syllable aware)
# - Auto-detect vocab mode (NFD Jamo / Compatibility Jamo / Hangul syllable)
# - Blank-posterior VAD (fallback no-VAD nếu crop quá ngắn)
# - Viterbi-local CTC per-unit confidence; optional time priors (TextGrid)
# - Lenient gate cho câu ngắn (≤2 âm tiết) nhưng có guard length-mismatch
# - Decode có temporal-bias theo thứ tự ref để giữ trật tự token ở câu dài
# - Tensification (받침→경음화) option cho compat
# - Unified LOGIT_TEMP/thresholds với w2v2_forced_scoring.py
# Python 3.9+

import os
import logging
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import unicodedata as ud
import torch.nn as nn

_log = logging.getLogger(__name__)
if not _log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

# ============ Config ============
MODEL_NAME = os.getenv("W2V_MODEL", "Kkonjeong/wav2vec2-base-korean")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR = 16000

# Thống nhất thông số với scoring file
VAD_PAD_MS = float(os.getenv("W2V_VAD_PAD_MS", "520.0"))
BLANK_VAD_THRESH = float(os.getenv("W2V_BLANK_VAD_THRESH", "0.8"))
LOGIT_TEMP = float(os.getenv("W2V_LOGIT_TEMP", "0.9"))            # dùng chung (decode/gate)
CONF_LOGIT_TEMP = float(os.getenv("W2V_CONF_LOGIT_TEMP", "0.8"))   # RIÊNG cho confidences

# Time-prior bias (log-space penalty khi frame nằm ngoài cửa sổ prior)
PRIOR_OUT_PENALTY = float(os.getenv("W2V_PRIOR_OUT_PENALTY", "12.0"))
PRIOR_EDGE_PAD_SEC = float(os.getenv("W2V_PRIOR_EDGE_PAD_SEC", "0.05"))
FORCED_MEAN_THR = float(os.getenv("W2V_FORCED_MEAN_THR", "0.25"))
FORCED_MAX_THR  = float(os.getenv("W2V_FORCED_MAX_THR",  "0.45"))

W2V_CONF_ALT_WEIGHT = float(os.getenv("W2V_CONF_ALT_WEIGHT",  "2.5"))
W2V_NB_MIN = float(os.getenv("W2V_NB_MIN",  "0.20"))
W2V_PREEMPH = float(os.getenv("W2V_PREEMPH",  "0.97"))

# Decode sắc nét, cộng với temporal bias giữ trật tự
DECODE_TEMP = float(os.getenv("W2V_DECODE_TEMP", "0.75"))
BLANK_SUPPRESS = float(os.getenv("W2V_BLANK_SUPPRESS", "0.50"))
VAD_USE_DECODE = int(os.getenv("W2V_VAD_USE_DECODE", "1"))
VAD_THR_DECODE = float(os.getenv("W2V_VAD_THR_DECODE", "0.55"))
VAD_PAD_MS_DECODE = float(os.getenv("W2V_VAD_PAD_MS_DECODE", "540.0"))

# Frame filtering cho confidences
CONF_FRAME_DELTA = float(os.getenv("W2V_CONF_FRAME_DELTA", "0.03"))  # p_ref - p_blank tối thiểu
CONF_TOP_P = float(os.getenv("W2V_CONF_TOP_P", "0.5"))               # chọn top-P frame theo p_ref (per unit)

# --- class/position-specific filters ---
CONF_TOPP_C_ONSET = float(os.getenv("W2V_CONF_TOPP_C_ONSET", "0.50"))
CONF_TOPP_C_CODA  = float(os.getenv("W2V_CONF_TOPP_C_CODA",  "0.30"))
CONF_TOPP_V       = float(os.getenv("W2V_CONF_TOPP_V",       "0.995"))

CONF_DELTA_C_ONSET = float(os.getenv("W2V_CONF_DELTA_C_ONSET", "0.05"))
CONF_DELTA_C_CODA  = float(os.getenv("W2V_CONF_DELTA_C_CODA",  "0.02"))
CONF_DELTA_V       = float(os.getenv("W2V_CONF_DELTA_V",       "-0.06"))

# competitor set size theo lớp
CONF_TOPK_COMP_V = int(os.getenv("W2V_CONF_TOPK_COMP_V", "1"))  # vowel: cạnh tranh hẹp
CONF_TOPK_COMP_C = int(os.getenv("W2V_CONF_TOPK_COMP_C", "1"))  # chỉ lấy 1 đối thủ khó nhất cho phụ âm

# floors theo vị trí (đặc biệt nâng onset như yêu cầu)
CONF_ONSET_FLOOR = float(os.getenv("W2V_CONF_ONSET_FLOOR", "0.80"))
CONF_VOWEL_FLOOR = float(os.getenv("W2V_CONF_VOWEL_FLOOR", "0.82"))
CONF_CODA_FLOOR  = float(os.getenv("W2V_CONF_CODA_FLOOR",  "0.80"))

W2V_CONF_ALT_WEIGHT_CODA = float(os.getenv("W2V_CONF_ALT_WEIGHT_CODA", "4.0"))  # ally weight cho codas
CONF_CODA_CAP = float(os.getenv("W2V_CONF_CODA_CAP", "0.96"))                   # hard cap để tránh 1.00
CODA_TAIL_KEEP = float(os.getenv("W2V_CODA_TAIL_KEEP", "0.90"))                 # giữ đoạn đuôi cho coda

END_VOWEL_TAIL_KEEP = float(os.getenv("W2V_END_VOWEL_TAIL_KEEP", "0.68"))

# Tensification công tắc
APPLY_TENSIFICATION = int(os.getenv("W2V_APPLY_TENSIFICATION", "1"))

# Globals (lazy)
_processor: Optional[Wav2Vec2Processor] = None
_model: Optional[Wav2Vec2ForCTC] = None
_blank_id: int = 0
_MODE: Optional[str] = None   # "nfd_jamo" | "compat_jamo" | "syllable"
_ID2TOK: Optional[Dict[int, str]] = None

# ===== Cấu hình ngôn ngữ (bộ lân cận, cluster nhận thức v.v.) =====
CODA_CLUSTER_TO_BASE = {
    0x11AA: 0x11A8,  # ᆪ(gs) → ᆨ
    0x11AC: 0x11AB,  # ᆬ(nj) → ᆫ
    0x11AD: 0x11AB,  # ᆭ(nh) → ᆫ
    0x11B0: 0x11AF,  # ᆰ(lg) → ᆯ
    0x11B1: 0x11AF,  # ᆱ(lm) → ᆯ
    0x11B2: 0x11AF,  # ᆲ(lb) → ᆯ
    0x11B3: 0x11AF,  # ᆳ(ls) → ᆯ
    0x11B4: 0x11AF,  # ᆴ(lt) → ᆯ
    0x11B5: 0x11AF,  # ᆵ(lp) → ᆯ
    0x11B6: 0x11AF,  # ᆶ(lh) → ᆯ
    0x11B9: 0x11B8,  # ᆹ(bs) → ᆸ
}

_VOWEL_NEIGHBORS = {
    "ㅏ": ["ㅏ","ㅐ","ㅑ"],     "ㅓ": ["ㅓ","ㅔ","ㅕ"],     "ㅗ": ["ㅗ","ㅘ","ㅚ","ㅛ"],
    "ㅜ": ["ㅜ","ㅝ","ㅟ","ㅠ"], "ㅡ": ["ㅡ","ㅢ"],          "ㅣ": ["ㅣ","ㅖ","ㅒ"],
    "ㅐ": ["ㅐ","ㅔ","ㅏ"],     "ㅔ": ["ㅔ","ㅐ","ㅓ"],     "ㅚ": ["ㅚ","ㅗ","ㅘ"],
    "ㅟ": ["ㅟ","ㅜ","ㅝ"],     "ㅘ": ["ㅘ","ㅗ","ㅚ"],     "ㅝ": ["ㅝ","ㅜ","ㅟ"],
}

# (CHỈNH) thu hẹp đối thủ coda ㅅ/ㅆ để giảm nhiễu từ ㄷ/ㅌ
_CODA_NEIGH_COMPAT = {
    "ㄱ": ["ㄱ","ㄲ","ㅋ","ㅅ"],  "ㄲ": ["ㄱ","ㄲ","ㅋ","ㅅ"],  "ㅋ": ["ㅋ","ㄱ","ㄲ"],
    "ㄴ": ["ㄴ","ㄷ","ㅌ"],       "ㄷ": ["ㄷ","ㅌ","ㅅ"],      "ㅌ": ["ㅌ","ㄷ"],
    "ㄹ": ["ㄹ","ㄴ"],           "ㅁ": ["ㅁ","ㅂ","ㅍ"],      "ㅂ": ["ㅂ","ㅍ","ㅅ"],
    "ㅍ": ["ㅍ","ㅂ"],           "ㅅ": ["ㅅ","ㅆ"],            "ㅆ": ["ㅆ","ㅅ"],
    "ㅇ": ["ㅇ","ㅎ"],           "ㅎ": ["ㅎ","ㅇ"],           "ㅈ": ["ㅈ","ㅊ","ㄷ"],
    "ㅊ": ["ㅊ","ㅈ","ㅌ"],      "ㅉ": ["ㅉ","ㅈ","ㅊ"]
}

# Vowel ally groups (merger forgiveness)
_VOWEL_ALLIES = {
    "ㅐ": ["ㅔ"],
    "ㅔ": ["ㅐ"],
    "ㅛ": ["ㅗ"], "ㅠ": ["ㅜ"], "ㅕ": ["ㅓ"], "ㅑ": ["ㅏ"],
    "ㅚ": ["ㅙ","ㅞ"], "ㅙ": ["ㅚ","ㅞ"], "ㅞ": ["ㅚ","ㅙ"],
}

# Compat sets
_COMPAT_CONS = set("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
_COMPAT_VOWS = set("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")

# coda obstruents (받침 gây căng hoá) & mapping lenis→tense
_CODA_OBSTRUENTS = set("ㄱㄲㅋㄷㅌㅂㅍㅅㅆㅈㅊ")
_LENIS_TO_TENSE = {"ㄱ":"ㄲ", "ㄷ":"ㄸ", "ㅂ":"ㅃ", "ㅅ":"ㅆ", "ㅈ":"ㅉ"}

# map compat -> NFD jamo
_COMPAT2CHO = { "ㄱ":"ᄀ","ㄲ":"ᄁ","ㄴ":"ᄂ","ㄷ":"ᄃ","ㄸ":"ᄄ","ㄹ":"ᄅ","ㅁ":"ᄆ","ㅂ":"ᄇ","ㅃ":"ᄈ","ㅅ":"ᄉ","ㅆ":"ᄊ","ㅇ":"ᄋ","ㅈ":"ᄌ","ㅉ":"ᄍ","ㅊ":"ᄎ","ㅋ":"ᄏ","ㅌ":"ᄐ","ㅍ":"ᄑ","ㅎ":"ᄒ" }
_COMPAT2JUN = { "ㅏ":"ᅡ","ㅐ":"ᅢ","ㅑ":"ᅣ","ㅒ":"ᅤ","ㅓ":"ᅥ","ㅔ":"ᅦ","ㅕ":"ᅧ","ㅖ":"ᅨ","ㅗ":"ᅩ","ㅘ":"ᅪ","ㅙ":"ᅫ","ㅚ":"ᅬ","ㅛ":"ᅭ","ㅜ":"ᅮ","ㅝ":"ᅯ","ㅞ":"ᅰ","ㅟ":"ᅱ","ㅠ":"ᅲ","ㅡ":"ᅳ","ㅢ":"ᅴ","ㅣ":"ᅵ" }
_COMPAT2JON = { "ㄱ":"ᆨ","ㄲ":"ᆩ","ㄳ":"ᆪ","ㄴ":"ᆫ","ㄵ":"ᆬ","ㄶ":"ᆭ","ㄷ":"ᆮ","ㄹ":"ᆯ","ㄺ":"ᆰ","ㄻ":"ᆱ","ㄼ":"ᆲ","ㄽ":"ᆳ","ㄾ":"ᆴ","ㄿ":"ᆵ","ㅀ":"ᆶ","ㅁ":"ᆷ","ㅂ":"ᆸ","ㅄ":"ᆹ","ㅅ":"ᆺ","ㅆ":"ᆻ","ㅇ":"ᆼ","ㅈ":"ᆽ","ㅊ":"ᆾ","ㅋ":"ᆿ","ㅌ":"ᇀ","ㅍ":"ᇁ","ㅎ":"ᇂ" }

def _is_space_or_punct(ch: str) -> bool:
    if not ch: 
        return False
    cat = ud.category(ch)
    # Z*: Separator (space, NBSP, v.v.); P*: Punctuation
    return cat.startswith('Z') or cat.startswith('P')

def _strip_space_punct(s: str) -> str:
    # bỏ toàn bộ ký tự phân tách & dấu câu (kể cả Unicode)
    return "".join(ch for ch in s if not _is_space_or_punct(ch))

# ===== Compat mapping đa âm tiết & tensification =====
def _compat_tokens_to_nfd_with_coda_guess(tokens: List[str]) -> str:
    """
    Gom toàn chuỗi compat theo pattern lặp CV(C?)|CV(C?)|...
    Trả về NFD jamo (choseong/jungseong/jongseong) đã gán đúng onset/vowel/coda theo âm tiết.
    """
    if not tokens:
        return ""

    # (đai an toàn) chỉ giữ đúng Jamo compat
    tokens = [t for t in tokens if (t in _COMPAT_CONS or t in _COMPAT_VOWS)]

    out = []
    i, n = 0, len(tokens)

    def as_cho(c): return _COMPAT2CHO.get(c, "")
    def as_jun(v): return _COMPAT2JUN.get(v, "")
    def as_jon(c): return _COMPAT2JON.get(c, "")

    while i < n:
        # bỏ qua rác nếu còn (phòng xa)
        if tokens[i] not in _COMPAT_CONS and tokens[i] not in _COMPAT_VOWS:
            i += 1
            continue

        # onset (tuỳ)
        cho = ""
        if tokens[i] in _COMPAT_CONS:
            cho = as_cho(tokens[i]); i += 1

        # vowel (bắt buộc để thành âm tiết)
        if i < n and tokens[i] in _COMPAT_VOWS:
            jun = as_jun(tokens[i]); i += 1
        else:
            # thiếu vowel → xuất riêng onset nếu có rồi tiếp
            if cho: out.append(cho)
            continue

        # coda (tuỳ) — logic: nếu sau V là C và sau nữa là V → C là onset của âm tiết sau
        jon = ""
        if i < n and tokens[i] in _COMPAT_CONS:
            if (i + 1 < n) and (tokens[i + 1] in _COMPAT_VOWS):
                pass  # C là onset syllable tiếp theo
            else:
                jon = as_jon(tokens[i]); i += 1

        out.append(cho + jun + jon)

    return "".join(out)

def _apply_tensification_compat(tokens: List[str]) -> List[str]:
    """
    Quy tắc đơn giản (word-internal):
    ... V C1 C2 V ...  với C1 ∈ CODA_OBSTRUENTS và C2 ∈ {ㄱ,ㄷ,ㅂ,ㅅ,ㅈ}
    => thay C2 bằng ssang tương ứng (ㄲ,ㄸ,ㅃ,ㅆ,ㅉ).
    """
    if len(tokens) < 4:
        return tokens[:]
    out = tokens[:]
    for i in range(len(tokens) - 3):
        V1, C1, C2, V2 = tokens[i], tokens[i+1], tokens[i+2], tokens[i+3]
        if (V1 in _COMPAT_VOWS and
            C1 not in _COMPAT_VOWS and C2 not in _COMPAT_VOWS and
            V2 in _COMPAT_VOWS and
            C1 in _CODA_OBSTRUENTS and C2 in _LENIS_TO_TENSE):
            out[i+2] = _LENIS_TO_TENSE[C2]
    return out

# ===== Vocab detection & mapping =====
def _to_nfd_jamo_only(s: str) -> str:
    s = ud.normalize("NFKD", s)
    return "".join(ch for ch in s if 0x1100 <= ord(ch) <= 0x11FF)

def _vocab_mode(vocab: Dict[str, int]) -> str:
    has_nfd = ("ᄋ" in vocab) or ("ᄀ" in vocab)
    has_compat = ("ㅇ" in vocab) or ("ㄱ" in vocab) or ("ㅏ" in vocab)
    if has_nfd: return "nfd_jamo"
    if has_compat: return "compat_jamo"
    return "syllable"

def _map_ref_to_compat(nfd: str) -> str:
    """NFD → compat, có hỗ trợ tense onset (ᄁ/ᄄ/ᄈ/ᄊ/ᄍ) và tense coda (ᆩ/ᆻ)."""
    out = []
    for ch in nfd:
        cp = ord(ch)
        if 0x1100 <= cp <= 0x1112:  # choseong
            base_map = {
                0x1100: 0x3131, 0x1101: 0x3132, 0x1102: 0x3134, 0x1103: 0x3137,
                0x1104: 0x3138, 0x1105: 0x3139, 0x1106: 0x3141, 0x1107: 0x3142,
                0x1108: 0x3143, 0x1109: 0x3145, 0x110A: 0x3146, 0x110B: 0x3147,
                0x110C: 0x3148, 0x110D: 0x3149, 0x110E: 0x314A, 0x110F: 0x314B,
                0x1110: 0x314C, 0x1111: 0x314D, 0x1112: 0x314E,
            }
            out.append(chr(base_map.get(cp, cp)))
        elif 0x1161 <= cp <= 0x1175:  # jung
            out.append(chr(cp - 0x1161 + 0x314F))
        elif 0x11A8 <= cp <= 0x11C2:  # jong
            jong_map = {
                0x11A8: 0x3131, 0x11A9: 0x3132, 0x11AB: 0x3134, 0x11AE: 0x3137,
                0x11AF: 0x3139, 0x11B7: 0x3141, 0x11B8: 0x3142, 0x11BA: 0x3145,
                0x11BB: 0x3146, 0x11BC: 0x3147, 0x11BD: 0x3148, 0x11BE: 0x314A,
                0x11BF: 0x314B, 0x11C0: 0x314C, 0x11C1: 0x314D, 0x11C2: 0x314E,
            }
            out.append(chr(jong_map.get(cp, cp)))
    return "".join(out)

def _reduce_coda_clusters(nfd: str) -> str:
    """Giảm jong chồng về phụ âm gốc để khớp vocab compat/NFD Jamo."""
    out = []
    for ch in nfd:
        cp = ord(ch)
        if 0x11A8 <= cp <= 0x11C2 and cp in CODA_CLUSTER_TO_BASE:
            out.append(chr(CODA_CLUSTER_TO_BASE[cp]))
        else:
            out.append(ch)
    return "".join(out)

def _ref_to_vocab_tokens(ref_text: str, vocab: Dict[str, int]) -> List[str]:
    # lọc dấu cách & dấu câu khỏi ref
    ref_text = _strip_space_punct(ref_text)

    nfd = _to_nfd_jamo_only(ref_text)
    nfd = _reduce_coda_clusters(nfd)
    if _MODE == "nfd_jamo":
        return [ch for ch in nfd if ch in vocab]
    elif _MODE == "compat_jamo":
        mapped = _map_ref_to_compat(nfd)   # produce compat jamo
        toks = [ch for ch in mapped if ch in vocab]
        if APPLY_TENSIFICATION:
            toks = _apply_tensification_compat(toks)
        return [ch for ch in toks if ch in vocab]
    else:
        s = ud.normalize("NFC", ref_text)
        return [ch for ch in s if (0xAC00 <= ord(ch) <= 0xD7A3) and (ch in vocab)]

# ===== (NEW) Helpers giữ separators cho HIỂN THỊ =====
def _compat_string_to_nfd_preserve_separators(s: str) -> str:
    """
    Biến chuỗi compat-jamo (có thể xen kẽ space/punct) thành NFD,
    nhưng GIỮ NGUYÊN các dấu cách / dấu câu ở đúng vị trí.
    """
    if not s:
        return ""
    out_parts: List[str] = []
    buf: List[str] = []
    def flush_buf():
        if buf:
            out_parts.append(_compat_tokens_to_nfd_with_coda_guess(buf))
            buf.clear()
    for ch in s:
        if ch in _COMPAT_CONS or ch in _COMPAT_VOWS:
            buf.append(ch)
        elif _is_space_or_punct(ch):
            flush_buf()
            out_parts.append(ch)  # giữ nguyên separator
        else:
            # ký tự khác vẫn giữ để hiển thị
            flush_buf()
            out_parts.append(ch)
    flush_buf()
    return "".join(out_parts)

def _to_display_nfd(s: str) -> str:
    """
    Hiển thị NFD: giữ Jamo + dấu cách/dấu câu.
    (Không cắt bỏ separators như luồng chấm điểm.)
    """
    if not s:
        return ""
    s_nfd = ud.normalize("NFKD", s)
    out = []
    for ch in s_nfd:
        cp = ord(ch)
        if _is_space_or_punct(ch) or (0x1100 <= cp <= 0x11FF):
            out.append(ch)
        # Nếu muốn giữ toàn bộ ký tự khác (emoji/Latin), bỏ comment dòng dưới:
        # else:
        #     out.append(ch)
    return "".join(out)

# ===== CER utils =====
def _levenshtein_distance(a: str, b: str) -> int:
    if a == b: return 0
    if len(a) < len(b): a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            insert = current[j-1] + 1
            delete = previous[j] + 1
            replace = previous[j-1] + (ca != cb)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]

def _cer(a: str, b: str) -> float:
    if not b:
        return 1.0 if a else 0.0
    return _levenshtein_distance(a, b) / max(1, len(b))

def _cer_skeleton(nfd_a: str, nfd_b: str) -> float:
    """Chỉ so onset+vowel (bỏ coda) để nới nhẹ câu rất ngắn."""
    a = "".join(ch for ch in nfd_a if (0x1100 <= ord(ch) <= 0x1112) or (0x1161 <= ord(ch) <= 0x1175))
    b = "".join(ch for ch in nfd_b if (0x1100 <= ord(ch) <= 0x1112) or (0x1161 <= ord(ch) <= 0x1175))
    return _cer(a, b)

# ===== Core init & audio pre =====
def _ensure_sr(waveform: torch.Tensor, sr: int) -> torch.Tensor:
    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

    # peak normalize để làm rõ biên độ
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak * 0.95

    # pre-emphasis (giúp onset rõ hơn)
    pre = float(os.getenv("W2V_PREEMPH", "0.0"))
    if pre and pre > 0.0:
        x = waveform
        x_shift = torch.nn.functional.pad(x, (1,0))[:,:-1]
        waveform = x - pre * x_shift

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    # pad nhẹ với đoạn quá ngắn
    sr_eff = TARGET_SR
    if waveform.shape[-1] < int(0.6 * sr_eff):
        pad = torch.zeros(1, int(0.2 * sr_eff), device=waveform.device, dtype=waveform.dtype)
        waveform = torch.cat([pad, waveform, pad], dim=-1)

    return waveform

def _init():
    """Lazy load model/processor và xác định vocab mode/blank."""
    global _processor, _model, _blank_id, _MODE, _ID2TOK
    if _processor is None or _model is None:
        _log.info(f"Loading W2V2 model for CTC: {MODEL_NAME}")
        _processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        _model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(DEVICE)
        _model.eval()
        tok = _processor.tokenizer
        _blank = getattr(tok, "pad_token_id", None)
        if _blank is None:
            vocab = tok.get_vocab()
            _blank = vocab.get("<pad>", 0)
        _blank_id = int(_blank)
        vocab = tok.get_vocab()
        _MODE = _vocab_mode(vocab)
        _ID2TOK = {v: k for k, v in vocab.items()}
        _log.info(f"[CTC] vocab_mode={_MODE}, blank_id={_blank_id}, vocab_size={len(_ID2TOK)}")

def _log_softmax(logits: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
    if temp != 1.0:
        logits = logits / max(1e-6, temp)
    return torch.log_softmax(logits, dim=-1)

def _blank_posterior_vad(logprobs: torch.Tensor, blank_id: int, thresh: float = 0.8, pad_ms: float = 250.0) -> slice:
    """Crop theo ngưỡng posterior của blank, có pad hai đầu."""
    with torch.no_grad():
        blank_post = logprobs.exp()[:, blank_id]
        non_blank = (blank_post < thresh).nonzero(as_tuple=False).squeeze(-1)
        if non_blank.numel() == 0:
            return slice(0, logprobs.shape[0])  # không thấy tiếng → giữ nguyên
        start = non_blank[0].item()
        end = non_blank[-1].item() + 1
        pad_frames = int(round(pad_ms / 20.0))  # ~20ms/frame
        start = max(0, start - pad_frames)
        end = min(logprobs.shape[0], end + pad_frames)
        return slice(start, end)

# ===== Best-path decoders (cứu cánh theo mức độ) =====
def _best_path_decode_nb_thresh(logprobs: torch.Tensor, vocab_id2tok: Dict[int, str],
                                nonblank_min: float = 0.30) -> str:
    out = []
    prev = None
    post = logprobs.exp()
    for t in range(post.shape[0]):
        p_t = post[t]
        p_blank = p_t[_blank_id]
        p_nb = p_t.clone(); p_nb[_blank_id] = 0.0
        i = int(p_nb.argmax().item())
        if p_nb[i] >= nonblank_min and p_nb[i] > p_blank:
            if i != prev:
                out.append(vocab_id2tok.get(i, ""))
            prev = i
        else:
            prev = _blank_id
    return "".join(out)

def _best_path_decode_nb_hys(logprobs, vocab_id2tok, nonblank_min=0.15, delta=0.02, min_run=2):
    post = logprobs.exp()
    out = []
    prev_tok = None
    run_tok, run_len = None, 0
    for t in range(post.shape[0]):
        p = post[t]
        p_blank = float(p[_blank_id])
        p_nb = p.clone(); p_nb[_blank_id] = 0.0
        i = int(p_nb.argmax().item())
        p_i = float(p_nb[i])
        if p_i >= nonblank_min and (p_i - p_blank) >= delta:
            if run_tok == i: run_len += 1
            else: run_tok, run_len = i, 1
        else:
            if run_tok is not None and run_len >= min_run:
                if run_tok != prev_tok:
                    out.append(vocab_id2tok.get(run_tok, ""))
                prev_tok = run_tok
            run_tok, run_len = None, 0
    if run_tok is not None and run_len >= min_run:
        if run_tok != prev_tok:
            out.append(vocab_id2tok.get(run_tok, ""))
    return "".join(out)

def _best_path_decode_no_blank(logprobs: torch.Tensor, vocab_id2tok: Dict[int,str],
                               banned: Optional[List[int]] = None) -> str:
    """Chọn argmax nhưng tắt blank/specials; collapse repeats."""
    post = logprobs.exp()
    V = post.shape[1]
    out, prev = [], None
    banned = set(banned or []); banned.add(_blank_id)
    mask = torch.ones(V, dtype=torch.bool, device=post.device)
    for b in banned:
        if 0 <= b < V: mask[b] = False
    for t in range(post.shape[0]):
        p = post[t]
        i = torch.argmax(torch.where(mask, p, torch.tensor(-1e9, device=p.device))).item()
        if i != prev:
            tok = vocab_id2tok.get(i, "")
            if tok not in ("", "<pad>", "<s>", "</s>"):
                out.append(tok)
        prev = i
    return "".join(out)

# ===== Confusable sets (only for Jamo modes) =====
CONFUSABLES: Dict[str, List[str]] = {
    "ᄂ": ["ᄅ"], "ᄅ": ["ᄂ"],
    "ᄇ": ["ᄑ"], "ᄑ": ["ᄇ"],
    "ᄃ": ["ᄐ", "ᄌ", "ᄎ"], "ᄐ": ["ᄃ", "ᄌ", "ᄎ"],
    "ᄌ": ["ᄃ", "ᄐ", "ᄎ"], "ᄎ": ["ᄃ", "ᄐ", "ᄌ"],
    "ᆪ": ["ᆨ", "ᆺ"], "ᆬ": ["ᆫ", "ᆽ"], "ᆭ": ["ᆫ", "ᇂ"],
    "ᆰ": ["ᆨ"], "ᆱ": ["ᆷ"], "ᆲ": ["ᆸ"], "ᆳ": ["ᆯ", "ᆺ"],
    "ᆴ": ["ᆯ"], "ᆵ": ["ᆸ"], "ᆶ": ["ᆯ"],
}

def _tok_class(t: str) -> str:
    """Phân lớp token (vowel/consonant theo mode)."""
    if not t or len(t) != 1:
        return "x"
    cp = ord(t)
    if _MODE == "compat_jamo":
        if t in _COMPAT_VOWS: return "v"
        if t in _COMPAT_CONS: return "c"
        return "x"
    else:
        if 0x1161 <= cp <= 0x1175: return "v"
        if 0x1100 <= cp <= 0x1112: return "c_ini"
        if 0x11A8 <= cp <= 0x11C2: return "c_coda"
        return "x"

def _compat_roles_for_ref_ids(ref_ids: List[int], id2tok: Dict[int, str]) -> List[str]:
    """
    Suy vai trò 'onset'|'vowel'|'coda' cho compat-jamo theo mẫu CV(C?)|CV(C?)|...
    (để dùng khi tính confidence/competitors).
    """
    roles: List[str] = []
    i, n = 0, len(ref_ids)

    def tok(k):
        t = id2tok.get(ref_ids[k], "")
        return t if t and len(t) == 1 else ""

    while i < n:
        t0 = tok(i)
        if t0 in _COMPAT_CONS:
            roles.append("onset"); i += 1
            if i >= n: break
            t1 = tok(i)
            if t1 in _COMPAT_VOWS:
                roles.append("vowel"); i += 1
                if i < n and tok(i) in _COMPAT_CONS:
                    roles.append("coda"); i += 1
            else:
                roles.append("vowel"); i += 1
        elif t0 in _COMPAT_VOWS:
            roles.append("vowel"); i += 1
        else:
            roles.append("coda"); i += 1

    roles = roles[:n]
    while len(roles) < n:
        roles.append("coda")
    return roles

def _expand_ref_with_blanks(ref_ids: List[int], blank_id: int) -> List[int]:
    """Interleave blank để xây ext-path cho Viterbi no-skip."""
    out = []
    for tid in ref_ids:
        out.append(blank_id)
        out.append(tid)
    out.append(blank_id)
    return out

def _uniform_token_windows(T: int, n_tok: int, overlap: float = 0.15) -> List[Tuple[float, float]]:
    """Chia T frame thành n_tok cửa sổ đều, có overlap nhẹ để giảm cứng nhắc thời gian."""
    if n_tok <= 0 or T <= 0:
        return []
    base = T / float(n_tok)
    pad = max(0.0, min(0.49, overlap)) * base
    wins = []
    for k in range(n_tok):
        c = (k + 0.5) * base
        s = max(0.0, c - base/2 - pad)
        e = min(float(T), c + base/2 + pad)
        wins.append((s, e))
    return wins

# ===== Forced decode (no-skip) để "vớt" khi best-path quá ngắn =====
def _ref_forced_decode(logprobs_dec: torch.Tensor,
                       vocab: Dict[str,int],
                       ref_text_nfd: str,
                       use_priors: bool = True) -> str:
    ref_toks = _ref_to_vocab_tokens(ref_text_nfd, vocab)
    ref_ids  = [vocab[t] for t in ref_toks if t in vocab]
    if not ref_ids:
        return ""

    def _viterbi_no_skip(logprobs: torch.Tensor, ext: List[int], bias_ext=None):
        T, M = logprobs.shape[0], len(ext)
        dp = torch.full((T, M), -1e9, device=logprobs.device)
        bp = torch.full((T, M), -1, device=logprobs.device, dtype=torch.long)
        step0 = logprobs[0, ext]
        if bias_ext is not None: step0 = step0 + bias_ext[0]
        dp[0, 0] = step0[0]
        if M > 1: dp[0, 1] = step0[1]; bp[0, 1] = 0
        for t in range(1, T):
            stay = dp[t-1]
            prev = torch.cat([torch.tensor([-1e9], device=dp.device), dp[t-1, :-1]], dim=0)
            best, arg = torch.stack([stay, prev], 0).max(0)
            step = logprobs[t, ext]
            if bias_ext is not None: step = step + bias_ext[t]
            dp[t] = best + step; bp[t] = arg
        m = M-1 if M>1 else 0
        if M>1 and dp[T-1, M-2] > dp[T-1, M-1]: m = M-2
        m_path = torch.empty(T, dtype=torch.long, device=dp.device); m_path[-1] = m
        for t in range(T-2, -1, -1):
            move = bp[t+1, m]
            if move == 1: m -= 1
            m_path[t] = m
        return m_path

    # priors cơ bản (ngắn 2-3 jamo)
    T = logprobs_dec.shape[0]
    bias_ext = None
    if use_priors:
        if len(ref_ids) == 3: wins = [(0.00,0.35),(0.25,0.75),(0.65,1.00)]
        elif len(ref_ids) == 2: wins = [(0.00,0.40),(0.35,1.00)]
        else: wins = [(0.00,1.00)]
        priors = []
        for ws,we in wins:
            s = max(0,int(np.floor(ws*T))); e = min(T,int(np.ceil(we*T)))
            priors.append((float(s), float(e)))
        bias_ext = _build_bias_ext_for_priors(
            T, _expand_ref_with_blanks(ref_ids, _blank_id), _blank_id,
            priors_sec=priors, sec_per_frame=1.0, crop_start_sec=0.0,
            out_penalty=PRIOR_OUT_PENALTY, edge_pad_sec=0.0
        )

    ext = _expand_ref_with_blanks(ref_ids, _blank_id)
    m_path = _viterbi_no_skip(logprobs_dec, ext, bias_ext=bias_ext)

    id2tok = {v:k for k,v in vocab.items()}
    out = []
    for i, tid in enumerate(ref_ids):
        m_idx = 2*i + 1
        if (m_path == m_idx).any():
            tok = id2tok.get(tid, "")
            if tok and (not out or out[-1] != tok):
                out.append(tok)
    return "".join(out)

def _build_bias_ext_for_priors(T_crop: int, ext: List[int], blank_id: int,
                               priors_sec: Optional[List[Optional[Tuple[float,float]]]],
                               sec_per_frame: float, crop_start_sec: float,
                               out_penalty: float = PRIOR_OUT_PENALTY,
                               edge_pad_sec: float = PRIOR_EDGE_PAD_SEC) -> Optional[torch.Tensor]:
    """Tạo bias ext-path (log) phạt nặng khi frame rời cửa sổ prior."""
    if not priors_sec:
        return None
    M = len(ext)
    bias = torch.zeros((T_crop, M), dtype=torch.float32)
    for i in range((M-1)//2):
        prior = priors_sec[i] if i < len(priors_sec) else None
        m_idx = 2*i + 1
        if prior is None:
            continue
        start_sec = max(0.0, float(prior[0]) - edge_pad_sec)
        end_sec = max(start_sec, float(prior[1]) + edge_pad_sec)
        start_f = int(np.floor((start_sec - crop_start_sec) / max(1e-6, sec_per_frame)))
        end_f = int(np.ceil((end_sec - crop_start_sec) / max(1e-6, sec_per_frame)))
        if end_f <= 0 or start_f >= T_crop:
            bias[:, m_idx] -= out_penalty
        else:
            if start_f > 0: bias[:max(0,start_f), m_idx] -= out_penalty
            if end_f < T_crop: bias[max(0,end_f):, m_idx] -= out_penalty
    return bias.to(DEVICE)

# ===== Per-unit confidences (Viterbi local) =====
def _compat_token_nfd_role(tok: str) -> str:
    """Compat token → 'v' nếu nguyên âm, 'c' nếu phụ âm, 'x' nếu khác (helper cũ, ít dùng)."""
    if tok in _COMPAT_VOWS: return "v"
    if tok in _COMPAT_CONS: return "c"
    return "x"

def _forced_ctc_jamo_confidences(
    logprobs: torch.Tensor,
    ref_ids: List[int],
    blank_id: int,
    confusable_ids: Optional[Dict[int, List[int]]] = None,
    bias_ext: Optional[torch.Tensor] = None,
    ally_ids: Optional[Dict[int, List[int]]] = None,
) -> List[float]:
    """Tính confidence từng đơn vị ref bằng Viterbi no-skip và so tỷ lệ logit với competitors."""
    ext = _expand_ref_with_blanks(ref_ids, blank_id)

    def _viterbi_no_skip(lp: torch.Tensor, ext_ids: List[int], bias: Optional[torch.Tensor]):
        T, M = lp.shape[0], len(ext_ids)
        dp = torch.full((T, M), -1e9, device=lp.device)
        bp = torch.full((T, M), -1,  device=lp.device, dtype=torch.long)
        step0 = lp[0, ext_ids] + (bias[0] if bias is not None else 0)
        dp[0, 0] = step0[0]
        if M > 1: dp[0, 1] = step0[1]; bp[0, 1] = 0
        for t in range(1, T):
            stay = dp[t-1]
            prev = torch.cat([torch.tensor([-1e9], device=dp.device), dp[t-1, :-1]], dim=0)
            best, arg = torch.stack([stay, prev], 0).max(0)
            step = lp[t, ext_ids] + (bias[t] if bias is not None else 0)
            dp[t] = best + step; bp[t] = arg
        m = M-1 if M > 1 else 0
        if M > 1 and dp[T-1, M-2] > dp[T-1, M-1]: m = M-2
        path = torch.empty(T, dtype=torch.long, device=dp.device); path[-1] = m
        for t in range(T-2, -1, -1):
            m -= bp[t+1, m].item()
            path[t] = m
        return path

    m_path = _viterbi_no_skip(logprobs, ext, bias_ext)
    V = logprobs.shape[1]
    id2tok = {v:k for k,v in _processor.tokenizer.get_vocab().items()}

    # --> vai trò theo compat chuỗi (hoặc theo codepoint trong NFD)
    if _MODE == "compat_jamo":
        roles = _compat_roles_for_ref_ids(ref_ids, id2tok)
    else:
        roles = []
        for tid in ref_ids:
            t = id2tok[tid]; cls = _tok_class(t)
            if cls == "c_ini": roles.append("onset")
            elif cls == "v":   roles.append("vowel")
            elif cls == "c_coda": roles.append("coda")
            else: roles.append("coda")

    # build mask competitors theo class 1 lần
    class_masks: Dict[int, torch.Tensor] = {}
    for tid in ref_ids:
        tok_str = id2tok[tid]
        cls = _tok_class(tok_str)
        mask = torch.zeros(V, dtype=torch.bool, device=logprobs.device)
        for tok, tokid in _processor.tokenizer.get_vocab().items():
            if tok in ("<pad>", "<s>", "</s>") or not tok or len(tok) != 1:
                continue
            if _tok_class(tok).startswith('v') and cls.startswith('v'):
                if tokid != tid: mask[tokid] = True
            elif _MODE == "nfd_jamo":
                if _tok_class(tok) == cls and tokid != tid:
                    mask[tokid] = True
            else:
                if _tok_class(tok) == 'c' and cls in ('c','c_ini','c_coda') and tokid != tid:
                    mask[tokid] = True

        # tighten cho compat coda bằng lân cận
        if _MODE == "compat_jamo" and cls in ('c','c_coda'):
            base_c = tok_str
            neigh = _CODA_NEIGH_COMPAT.get(base_c, None)
            if neigh:
                keep_ids = []
                for n in neigh:
                    k_id = _processor.tokenizer.get_vocab().get(n)
                    if k_id is not None: keep_ids.append(k_id)
                tight = torch.zeros_like(mask)
                for k_id in keep_ids:
                    if 0 <= k_id < tight.shape[0] and k_id != tid:
                        tight[k_id] = True
                mask &= tight

        # loại allies (cluster variants)
        if ally_ids and tid in ally_ids:
            for a_id in ally_ids[tid]:
                if 0 <= a_id < mask.shape[0]:
                    mask[a_id] = False

        class_masks[tid] = mask

    confs: List[float] = []
    for i, tid in enumerate(ref_ids):
        # NOTE: dùng vai trò đã suy từ chuỗi
        pos = roles[i] if i < len(roles) else "coda"

        # khung thăm state m_idx
        m_idx = 2*i + 1
        mask_visit = (m_path == m_idx)

        # nếu chưa ghé state, chọn frame tốt nhất theo bias/likelihood
        if not mask_visit.any():
            if bias_ext is not None:
                best_t = int(torch.argmax(bias_ext[:, m_idx]).item())
            else:
                best_t = int(torch.argmax(logprobs[:, tid]).item())
            mask_visit = torch.zeros(logprobs.shape[0], dtype=torch.bool, device=logprobs.device)
            mask_visit[best_t] = True

        seg_log = logprobs[mask_visit]  # [Ti, V]
        if seg_log.shape[0] == 0:
            confs.append(0.0)
            continue

        # coda: lấy phần đuôi (giảm nhiễu đoạn chuyển tiếp)
        if pos == "coda" and seg_log.shape[0] >= 3:
            start = int(CODA_TAIL_KEEP * seg_log.shape[0])
            seg_log = seg_log[start:]

        # vowel cuối: lấy phần đuôi (giữ năng lượng rơi câu)
        if pos == "vowel" and seg_log.shape[0] >= 3:
            is_last_vowel = (i == len(ref_ids) - 1) or (i == len(ref_ids) - 2 and roles[i+1] != "vowel")
            if is_last_vowel:
                vstart = int(END_VOWEL_TAIL_KEEP * seg_log.shape[0])
                seg_log = seg_log[vstart:]

        seg = seg_log.exp()
        p_blank = seg[:, blank_id]
        p_ref   = seg[:, tid]

        # lọc khung theo vị trí
        if pos == "onset":
            delta = max(CONF_FRAME_DELTA, CONF_DELTA_C_ONSET)
            top_p = CONF_TOPP_C_ONSET
        elif pos == "vowel":
            delta = min(CONF_FRAME_DELTA, CONF_DELTA_V)
            top_p = CONF_TOPP_V
        else:
            delta = max(CONF_FRAME_DELTA, CONF_DELTA_C_CODA)
            top_p = CONF_TOPP_C_CODA

        keep = (p_ref - p_blank) >= delta
        if keep.any():
            seg_log = seg_log[keep]; seg = seg[keep]; p_blank = p_blank[keep]; p_ref = p_ref[keep]

        if seg.shape[0] > 1 and 0.1 <= top_p < 1.0:
            k = max(1, int(np.ceil(top_p * seg.shape[0])))
            top_idx = torch.topk(p_ref, k).indices
            seg_log = seg_log[top_idx]; seg = seg[top_idx]; p_blank = p_blank[top_idx]; p_ref = p_ref[top_idx]

        # competitors cùng lớp + thu hẹp hàng xóm cho nguyên âm
        cls_mask = class_masks[tid].clone()
        tok_str = id2tok[tid]
        if _tok_class(tok_str).startswith('v') and _MODE == "compat_jamo":
            neigh = _VOWEL_NEIGHBORS.get(tok_str, None)
            if neigh:
                keep_ids = [ _processor.tokenizer.get_vocab().get(n) for n in neigh if n in _processor.tokenizer.get_vocab() ]
                mask_neigh = torch.zeros_like(cls_mask)
                for k_id in keep_ids:
                    if k_id is not None and 0 <= k_id < cls_mask.shape[0]:
                        mask_neigh[k_id] = True
                cls_mask &= mask_neigh

        comp_logits = seg_log[:, cls_mask]
        if comp_logits.numel() == 0:
            ci = float((p_ref / (p_ref + p_blank)).mean().item())
        else:
            if pos == "coda":
                K = 2
                top_p = min(top_p, 0.30)
                delta = max(delta, 0.08)
            else:
                K = (CONF_TOPK_COMP_V if pos == "vowel" else CONF_TOPK_COMP_C)
            if K > 0 and comp_logits.shape[1] > K:
                comp_logits, _ = torch.topk(comp_logits, k=K, dim=1)
            from torch import logsumexp
            log_ref = seg_log[:, tid]
            # ally support (nếu có)
            if ally_ids and tid in ally_ids:
                ally_cols = [a for a in ally_ids[tid] if 0 <= a < seg_log.shape[1]]
                if ally_cols:
                    w = (W2V_CONF_ALT_WEIGHT_CODA if pos == "coda" else W2V_CONF_ALT_WEIGHT)
                    ally_log = seg_log[:, ally_cols] + np.log(max(1.0, w))
                    log_ref = logsumexp(torch.stack([log_ref, logsumexp(ally_log, dim=1)], dim=0), dim=0)

            log_comp = logsumexp(comp_logits, dim=1) if comp_logits.numel() else p_blank.log()
            margin = log_ref - log_comp

            # percentile pooling theo vị trí
            if pos == "vowel":
                Q, TAU, BIAS, GAMMA = 0.80, 0.24, 0.06, 0.90
            elif pos == "onset":
                Q, TAU, BIAS, GAMMA = 0.97, 0.20, 0.03, 0.92
            else:
                Q, TAU, BIAS, GAMMA = 0.93, 0.22, 0.06, 0.90

            k = max(1, int(np.ceil(Q * margin.shape[0])))
            topk = torch.topk(margin, k).values
            ci_raw = torch.sigmoid(topk / max(1e-6, TAU)).mean().item()
            ci_cal = (ci_raw - BIAS) / max(1e-6, (1.0 - BIAS))
            ci_cal = max(0.02, min(1.0, ci_cal))
            ci = ci_cal ** GAMMA

        # floors theo vị trí
        if pos == "onset":
            ci = max(ci, max(CONF_ONSET_FLOOR, 0.78))
        elif pos == "vowel":
            ci = max(ci, CONF_VOWEL_FLOOR)
        else:
            ci = max(ci, CONF_CODA_FLOOR)

        # boost nhẹ riêng cho coda ㄱ
        tok_str = id2tok[tid]
        if pos == "coda" and tok_str in ("ᆨ","ㄱ"):  # tuỳ mode
            ci = min(0.98, ci + 0.03)

        # (NEW) micro-boost cho coda tense sibilant (ᆻ/ㅆ)
        if pos == "coda" and tok_str in ("ᆻ","ㅆ"):
            ci = min(CONF_CODA_CAP, ci + 0.03)

        confs.append(ci)

    if len(ref_ids) >= 2 and len(confs) >= 2 and confs[1] >= 0.97 and confs[0] >= 0.80:
        confs[0] = min(0.92, confs[0] + 0.03)

    if not any(c > 0.0 for c in confs):
        _log.info("[conf_fallback] all-zero confs")
        return [0.0] * len(ref_ids)
    
    # synergy: onset nhích nếu vowel sau rất chắc
    if len(confs) >= 2:
        if confs[1] >= 0.97 and confs[0] >= 0.70:
            confs[0] = min(0.94, confs[0] + 0.06)
    return confs

def _ids_of(tokens, vocab):
    return [vocab[t] for t in tokens if t in vocab]

def _bias_decode_logits(logprobs_dec: torch.Tensor, vocab: Dict[str,int],
                        ref_text_nfd: str,
                        filler_penalty: float = 1.6,
                        inref_bonus: float = 0.8) -> torch.Tensor:
    """
    Áp bias decode: phạt 'ㅇ' nếu không thuộc ref; bonus cho token nằm trong ref;
    bonus nhẹ riêng cho nguyên âm để tránh chìm.
    """
    ref_toks = _ref_to_vocab_tokens(ref_text_nfd, vocab)
    ref_ids = _ids_of(ref_toks, vocab)
    out = logprobs_dec.clone()

    # bonus cho token trong ref
    if ref_ids:
        out[:, ref_ids] += inref_bonus
    # bonus nhẹ riêng cho nguyên âm
    for t in ref_toks:
        cp = ord(t) if t else 0
        if (_MODE == "compat_jamo" and t in _COMPAT_VOWS) or (0x1161 <= cp <= 0x1175):
            if t in vocab:
                out[:, vocab[t]] += 0.25
    # phạt filler ㅇ nếu không có trong ref
    filler = "ㅇ" if _MODE == "compat_jamo" else "ᄋ"
    if filler not in ref_toks and filler in vocab:
        out[:, vocab[filler]] -= filler_penalty
    # phạt specials
    for sp in ("<pad>", "<s>", "</s>"):
        if sp in vocab:
            out[:, vocab[sp]] -= filler_penalty
    return out

# ===== Public API =====
def ctc_char_confidences_from_waveform(waveform: torch.Tensor, sr: int, ref_text_nfd: str,
                                       blank_vad_thresh: float = None, vad_pad_ms: float = None,
                                       temp: float = None, use_confusables: bool = True,
                                       priors_sec: Optional[List[Optional[Tuple[float,float]]]] = None
                                       ) -> List[float]:
    """
    Tính confidence cho từng token ref (theo vocab mode hiện tại).
    - VAD dựa blank posterior (fallback no-VAD khi quá ngắn)
    - Viterbi local no-skip với optional priors
    - Competitors theo lớp + hàng xóm nguyên âm / coda
    """
    _init()
    Ttemp_in = (CONF_LOGIT_TEMP if temp is None else temp)
    Ttemp = min(1.0, float(Ttemp_in))
    _log.info("[conf_cfg] CONF_LOGIT_TEMP(eff)=%.3f TOP_P=%.2f DELTA=%.3f ALT_W=%.2f BLANK_THR=%.2f PAD_MS=%.1f MODE=%s",
              Ttemp, CONF_TOP_P, CONF_FRAME_DELTA, W2V_CONF_ALT_WEIGHT,
              (BLANK_VAD_THRESH if blank_vad_thresh is None else blank_vad_thresh),
              (VAD_PAD_MS if vad_pad_ms is None else vad_pad_ms), _MODE)

    blank_thr = BLANK_VAD_THRESH if blank_vad_thresh is None else blank_vad_thresh
    pad_ms = VAD_PAD_MS if vad_pad_ms is None else vad_pad_ms

    wav = _ensure_sr(waveform, sr)
    total_sec = float(wav.shape[-1]) / TARGET_SR
    with torch.inference_mode():
        inputs = _processor(wav.squeeze(0), sampling_rate=TARGET_SR, return_tensors="pt", padding=False)
        input_values = inputs.input_values.to(DEVICE)
        logits = _model(input_values).logits[0]
        logprobs_full = _log_softmax(logits, temp=Ttemp)

        # VAD crop
        vslice = _blank_posterior_vad(logprobs_full, _blank_id, thresh=blank_thr, pad_ms=pad_ms)
        logprobs = logprobs_full[vslice]
        T_full = logprobs_full.shape[0]
        sec_per_frame = total_sec / max(1, T_full)
        crop_start_sec = vslice.start * sec_per_frame

        # fallback nếu crop quá ngắn
        if logprobs.shape[0] < 6:
            logprobs = logprobs_full
            crop_start_sec = 0.0

        vocab = _processor.tokenizer.get_vocab()
        ref_tokens_all = _ref_to_vocab_tokens(ref_text_nfd, vocab)
        ref_ids = [vocab[t] for t in ref_tokens_all if t in vocab]

        # cảnh báo mismatch
        try:
            nfd_len_dbg = len(_to_nfd_jamo_only(_strip_space_punct(ref_text_nfd)))
            if _MODE == "compat_jamo" and nfd_len_dbg >= 4 and len(ref_ids) <= 3:
                _log.warning("[compat_ref_len] NFD=%d vs compat_ids=%d → khả năng thiếu âm tiết/coda, kiểm tra ref/vocab.",
                            nfd_len_dbg, len(ref_ids))
        except Exception:
            pass

        # align priors nếu có
        aligned_priors: List[Optional[Tuple[float,float]]] = None
        if priors_sec is not None and len(priors_sec) > 0:
            aligned_priors = priors_sec if len(priors_sec) == len(ref_ids) else [None] * len(ref_ids)

        if len(ref_ids) == 0:
            _log.warning("Reference tokens not found in vocab. mode=%s; hãy kiểm tra model/vocab.", _MODE)
            return [0.0] * max(1, len(ref_tokens_all))

        # confusable ids cho jamo modes
        use_confusables = use_confusables and (_MODE != "syllable")
        conf_ids: Dict[int, List[int]] = {}
        if use_confusables:
            for t_nfd, alts in CONFUSABLES.items():
                if not t_nfd or len(t_nfd) != 1: continue
                cp = ord(t_nfd)
                if not (0x1100 <= cp <= 0x1112):  # chỉ choseong
                    continue
                base_c = _nfd_choseong_to_compat(cp)
                if base_c and base_c in vocab:
                    base_id_c = vocab[base_c]
                    conf_ids.setdefault(base_id_c, [])
                    for a in alts:
                        if not a or len(a) != 1: continue
                        acp = ord(a)
                        if not (0x1100 <= acp <= 0x1112): continue
                        alt_c = _nfd_choseong_to_compat(acp)
                        if alt_c and alt_c in vocab:
                            conf_ids[base_id_c].append(vocab[alt_c])

        # ally map cho cluster coda (base ↔ clusters)
        reverse_clusters = {}
        for clus, base in CODA_CLUSTER_TO_BASE.items():
            reverse_clusters.setdefault(base, []).append(clus)

        def _add_alt(tcp_base, tcp_cluster_list):
            t_base = chr(tcp_base)
            if t_base in vocab:
                base_id = vocab[t_base]
                conf_ids.setdefault(base_id, [])
                for ccp in tcp_cluster_list:
                    t_clu = chr(ccp)
                    if t_clu in vocab:
                        conf_ids[base_id].append(vocab[t_clu])
            comp_base = _map_ref_to_compat(t_base)
            if comp_base and comp_base[0] in vocab:
                base_id_c = vocab[comp_base[0]]
                conf_ids.setdefault(base_id_c, [])
                for ccp in tcp_cluster_list:
                    comp_clu = _map_ref_to_compat(chr(ccp))
                    if comp_clu and comp_clu[0] in vocab:
                        conf_ids[base_id_c].append(vocab[comp_clu[0]])

        for tcp_base, lst in reverse_clusters.items():
            _add_alt(tcp_base, lst)

        if conf_ids:
            some = list(conf_ids.items())[:5]
            _log.info("[conf_alt] entries=%d sample_keys=%s", len(conf_ids), [k for k,_ in some])

        # priors heuristics cho chuỗi 2-3 token
        aligned_priors = aligned_priors or []
        if (priors_sec is None or len(priors_sec) == 0) and len(ref_ids) == 2:
            total_crop_sec = logprobs.shape[0] * sec_per_frame
            aligned_priors = [(0.00 * total_crop_sec, 0.28 * total_crop_sec),
                              (0.32 * total_crop_sec, 1.00 * total_crop_sec)]
        if (priors_sec is None or len(priors_sec) == 0) and len(ref_ids) == 3:
            total_crop_sec = logprobs.shape[0] * sec_per_frame
            aligned_priors = [(0.00 * total_crop_sec, 0.28 * total_crop_sec),
                              (0.18 * total_crop_sec, 0.92 * total_crop_sec),
                              (0.70 * total_crop_sec, 1.00 * total_crop_sec)]

        bias_ext = None
        if aligned_priors and any(p is not None for p in aligned_priors):
            pen = max(2.0, PRIOR_OUT_PENALTY * (0.9 / max(1e-6, Ttemp)))
            bias_ext = _build_bias_ext_for_priors(
                logprobs.shape[0], _expand_ref_with_blanks(ref_ids, _blank_id), _blank_id,
                aligned_priors, sec_per_frame, crop_start_sec,
                out_penalty=pen, edge_pad_sec=PRIOR_EDGE_PAD_SEC
            )

        # Ally map cho confidences (từ ref NFD raw để biết cluster thực sự)
        ally_ids: Dict[int, List[int]] = {}
        ref_nfd_raw = _to_nfd_jamo_only(ref_text_nfd)
        vocab = _processor.tokenizer.get_vocab()
        for ch in ref_nfd_raw:
            cp = ord(ch)
            if 0x11A8 <= cp <= 0x11C2 and cp in CODA_CLUSTER_TO_BASE:
                base_cp = CODA_CLUSTER_TO_BASE[cp]
                if _MODE == "compat_jamo":
                    base_tok = _map_ref_to_compat(chr(base_cp))
                    base_id = vocab.get(base_tok[0]) if base_tok else None
                    ally_tok = _map_ref_to_compat(chr(cp))
                    ally_id = vocab.get(ally_tok[0]) if ally_tok else None
                else:
                    base_id = vocab.get(chr(base_cp))
                    ally_id = vocab.get(chr(cp))
                if base_id is not None and ally_id is not None:
                    ally_ids.setdefault(base_id, [])
                    if ally_id not in ally_ids[base_id]:
                        ally_ids[base_id].append(ally_id)

        # --- Vowel allies (compat mode) ---
        if _MODE == "compat_jamo":
            vocab_map = _processor.tokenizer.get_vocab()
            id2tok = {v:k for k,v in vocab_map.items()}
            for tid in ref_ids:
                t = id2tok.get(tid, "")
                if t in _VOWEL_ALLIES:
                    alts = _VOWEL_ALLIES[t]
                    for a in alts:
                        aid = vocab_map.get(a, None)
                        if aid is not None:
                            ally_ids.setdefault(tid, [])
                            if aid not in ally_ids[tid]:
                                ally_ids[tid].append(aid)

        # --- (NEW) Coda allies cho sibilants: ㅆ/ᆻ được trợ lực từ ㅅ/ᆺ ---
        try:
            vocab_map = _processor.tokenizer.get_vocab()
            # compat
            sid_ssang = vocab_map.get("ㅆ")
            sid_lenis = vocab_map.get("ㅅ")
            # nfd-jong
            jid_ssang = vocab_map.get("ᆻ")
            jid_lenis = vocab_map.get("ᆺ")

            def _add_ally(base_id, ally_id):
                if base_id is not None and ally_id is not None:
                    ally_ids.setdefault(base_id, [])
                    if ally_id not in ally_ids[base_id]:
                        ally_ids[base_id].append(ally_id)

            _add_ally(sid_ssang, sid_lenis)
            _add_ally(jid_ssang, jid_lenis)
        except Exception:
            pass

        # temporal priors cho confidences, giống decode
        conf_bias_ext = None
        if _MODE == "compat_jamo":
            ref_toks = _ref_to_vocab_tokens(ref_text_nfd, vocab)
            ref_ids2 = [vocab[t] for t in ref_toks if t in vocab]
            if len(ref_ids2) >= 4:
                wins = _uniform_token_windows(logprobs.shape[0], len(ref_ids2), overlap=0.22)
                priors_sec2 = [(w[0] * sec_per_frame + crop_start_sec, w[1] * sec_per_frame + crop_start_sec) for w in wins]
                pen = max(2.0, PRIOR_OUT_PENALTY * (0.9 / max(1e-6, Ttemp)))
                conf_bias_ext = _build_bias_ext_for_priors(
                    logprobs.shape[0],
                    _expand_ref_with_blanks(ref_ids2, _blank_id),
                    _blank_id,
                    priors_sec2, sec_per_frame, crop_start_sec,
                    out_penalty=pen, edge_pad_sec=PRIOR_EDGE_PAD_SEC
                )

        confs = _forced_ctc_jamo_confidences(
            logprobs, ref_ids, _blank_id,
            confusable_ids=conf_ids, bias_ext=conf_bias_ext, ally_ids=ally_ids
        )

        # pad về độ dài ref_tokens_all để UI an toàn
        if len(confs) < len(ref_tokens_all):
            confs = confs + [0.0] * (len(ref_tokens_all) - len(confs))

        try:
            mean_conf = float(np.mean(confs)) if len(confs) > 0 else -1
            _log.info(f"[ctc_char_conf] mode={_MODE} T_full={T_full} T_crop={logprobs.shape[0]} "
                      f"len(ref_tokens)={len(ref_tokens_all)} len(ref_ids)={len(ref_ids)} mean_conf={mean_conf:.3f}")
        except Exception:
            pass

        return confs

def ctc_char_confidences(wav_path: str, ref_text_nfd: str, **kwargs) -> List[float]:
    wav, sr = torchaudio.load(wav_path)
    return ctc_char_confidences_from_waveform(wav, sr, ref_text_nfd, **kwargs)

def _best_path_decode(logprobs: torch.Tensor, vocab_id2tok: Dict[int, str]) -> str:
    post = logprobs.exp()
    nonblank_mean = float((1.0 - post[:, _blank_id]).mean().item())
    _log.info(f"[dbg] mean(1-P(blank)) = {nonblank_mean:.3f}")
    ids = logprobs.argmax(dim=-1).tolist()
    out = []; prev = None
    for i in ids:
        if i == _blank_id:
            prev = i; continue
        if i != prev:
            out.append(vocab_id2tok.get(i, ""))
        prev = i
    return "".join(out)

def _ctc_ref_nll_per_char(logprobs: torch.Tensor, vocab: Dict[str,int],
                          ref_text_nfd: str, blank_id: int) -> float:
    """CTC loss/char cho ref → dùng làm một nhánh gate."""
    tgt_tokens = _ref_to_vocab_tokens(ref_text_nfd, vocab)
    ref_ids = [vocab[ch] for ch in tgt_tokens if ch in vocab]
    if len(ref_ids) == 0 or logprobs.shape[0] == 0:
        return 10.0
    logp = logprobs.unsqueeze(1)  # (T,1,V)
    input_lengths = torch.tensor([logp.shape[0]], dtype=torch.long, device=logp.device)
    target_lengths = torch.tensor([len(ref_ids)], dtype=torch.long, device=logp.device)
    targets = torch.tensor(ref_ids, dtype=torch.long, device=logp.device)
    ctc_loss = nn.CTCLoss(blank=blank_id, reduction="sum", zero_infinity=True)
    loss = ctc_loss(logp, targets, input_lengths, target_lengths)
    return float(loss.item() / max(1, len(ref_ids)))

def _nfd_choseong_to_compat(cp: int) -> Optional[str]:
    ch = chr(cp)
    s = _map_ref_to_compat(ch)
    return s[0] if s else None

# ===== Gate tổng hợp =====
def ctc_gate_global_from_waveform(waveform: torch.Tensor, sr: int, ref_text_nfd: str,
                                  cer_threshold: float = 0.65,
                                  loss_char_threshold: float = 2.0,
                                  ref_nll_threshold: float = float(os.getenv("W2V_CTC_REF_NLL_THR", "2.2")),
                                  short_syll_cer_bonus: float = 0.05) -> Dict[str, float]:
    """
    Gate “lai”: pass nếu một trong các điều kiện:
      (A) CER + mean_neglogp đạt,
      (B) CTC-NLL/ref_char thấp,
      (C) Câu rất ngắn (≤2 âm tiết): skeleton CER đạt.
    Có "guard" length-mismatch để tránh pass sai khi ref dài nhưng compat tokens ngắn.
    """
    _init()
    wav = _ensure_sr(waveform, sr)
    with torch.inference_mode():
        inputs = _processor(wav.squeeze(0), sampling_rate=TARGET_SR, return_tensors="pt", padding=False)
        input_values = inputs.input_values.to(DEVICE)
        logits = _model(input_values).logits[0]

        # hai bản logprobs: decode (sắc) & confidences (ổn định)
        logprobs_decode = _log_softmax(logits, temp=DECODE_TEMP)
        logprobs_full   = _log_softmax(logits, temp=LOGIT_TEMP)

        # VAD cho decode hoặc full
        if VAD_USE_DECODE:
            sl = _blank_posterior_vad(logprobs_decode, _blank_id, thresh=VAD_THR_DECODE, pad_ms=VAD_PAD_MS_DECODE)
        else:
            sl = _blank_posterior_vad(logprobs_full, _blank_id, thresh=BLANK_VAD_THRESH, pad_ms=VAD_PAD_MS)

        logprobs_dec = logprobs_decode[sl]
        logprobs_conf = logprobs_full[sl]

        # suppress blank nhẹ cho decode
        if BLANK_SUPPRESS > 0:
            logprobs_dec = logprobs_dec.clone()
            logprobs_dec[:, _blank_id] -= BLANK_SUPPRESS

        # fallback khi crop quá ngắn / không đủ nonblank
        if logprobs_dec.shape[0] < 6:
            logprobs_dec = logprobs_decode
            logprobs_conf = logprobs_full
        nonblank_mean = float((1.0 - logprobs_dec.exp()[:, _blank_id]).mean().item())
        if logprobs_dec.shape[0] < 6 or nonblank_mean < 0.35:
            logprobs_dec  = logprobs_decode
            logprobs_conf = logprobs_full

        vocab = _processor.tokenizer.get_vocab()
        id2tok = {v: k for k, v in vocab.items()}

        # decode bias theo ref + bonus nguyên âm + phạt filler
        logprobs_dec_biased = _bias_decode_logits(
            logprobs_dec, vocab, ref_text_nfd,
            filler_penalty=float(os.getenv("W2V_FILLER_PEN", "1.6")),
            inref_bonus=float(os.getenv("W2V_INREF_BONUS", "0.8"))
        )

        # temporal-bias đều theo thứ tự ref (câu dài)
        if _MODE == "compat_jamo":
            ref_toks = _ref_to_vocab_tokens(ref_text_nfd, vocab)
            ref_ids  = [vocab[t] for t in ref_toks if t in vocab]
            if len(ref_ids) >= 4:
                wins = _uniform_token_windows(logprobs_dec_biased.shape[0], len(ref_ids), overlap=0.20)
                tbias = torch.zeros_like(logprobs_dec_biased)
                for k, tid in enumerate(ref_ids):
                    s, e = int(wins[k][0]), int(wins[k][1])
                    if 0 <= tid < tbias.shape[1] and s < e:
                        tbias[s:e, tid] += 0.35
                logprobs_dec_biased = logprobs_dec_biased + tbias

        # chuỗi cứu cánh decode
        hyp = _best_path_decode(logprobs_dec_biased, id2tok)
        if len(_to_nfd_jamo_only(hyp)) < len(_ref_to_vocab_tokens(ref_text_nfd, vocab)):
            hyp_forced = _ref_forced_decode(logprobs_dec_biased, vocab, ref_text_nfd, use_priors=True)
            if hyp_forced.strip():
                hyp = hyp_forced
        if not hyp.strip():
            hyp = _best_path_decode_no_blank(logprobs_dec_biased, id2tok, banned=None)
        if not hyp.strip():
            hyp = _best_path_decode_nb_thresh(logprobs_dec_biased, id2tok, nonblank_min=float(os.getenv("W2V_NB_MIN","0.10")))
        if not hyp.strip():
            hyp = _best_path_decode_nb_hys(logprobs_dec_biased, id2tok, nonblank_min=float(os.getenv("W2V_NB_MIN","0.10")), delta=0.00, min_run=1)
        if not hyp.strip():
            post = logprobs_dec_biased.exp()
            p = post.mean(0); p[_blank_id] = 0.0
            for sp in ("<pad>", "<s>", "</s>"):
                if sp in vocab: p[vocab[sp]] = 0.0
            i = int(torch.argmax(p).item()); hyp = id2tok.get(i, "") or hyp

        if len(_to_nfd_jamo_only(hyp)) < 2:
            hyp_forced = _ref_forced_decode(logprobs_dec_biased, vocab, ref_text_nfd, use_priors=True)
            if hyp_forced.strip():
                hyp = hyp_forced

        # === HIỂN THỊ: giữ bản thô (có separators) ===
        hyp_display = hyp
        print("hyp", hyp_display)

        # === SCORING: loại separators khỏi CER như trước ===
        hyp = hyp.strip()
        if _MODE == "compat_jamo":
            hyp_tokens = [ch for ch in hyp if (ch in _COMPAT_CONS or ch in _COMPAT_VOWS)]
            hyp_nfd = _compat_tokens_to_nfd_with_coda_guess(hyp_tokens)
            # HIỂN THỊ: giữ separators
            hyp_display_nfd = _compat_string_to_nfd_preserve_separators(hyp_display)
        else:
            hyp_nfd = _to_nfd_jamo_only(hyp)
            hyp_display_nfd = _to_display_nfd(hyp_display)

        print("hyp_nfd", hyp_nfd)

        # REF cho SCORING (lọc separators)
        ref_nfd = _to_nfd_jamo_only(_strip_space_punct(ref_text_nfd))
        ref_nfd = _reduce_coda_clusters(ref_nfd)
        # REF cho HIỂN THỊ (giữ separators)
        ref_display_nfd = _to_display_nfd(ref_text_nfd)

        cer = _cer(hyp_nfd, ref_nfd)
        cer_skel = _cer_skeleton(hyp_nfd, ref_nfd)

        # mean_neglogp: trung bình posterior của tokens ref (theo space hiện tại)
        ref_tokens_for_mean = _ref_to_vocab_tokens(ref_text_nfd, vocab)
        ref_ids_for_mean = [vocab[ch] for ch in ref_tokens_for_mean if ch in vocab]
        if len(ref_ids_for_mean) == 0 or logprobs_dec.shape[0] == 0:
            mean_neglogp = 10.0
        else:
            tok_post = logprobs_dec.exp().mean(0)
            pick = tok_post[ref_ids_for_mean].clamp_min(1e-6)
            mean_neglogp = float((-pick.log()).mean().item())

        ref_nll_pc = _ctc_ref_nll_per_char(logprobs_dec, vocab, ref_text_nfd, _blank_id)

        # nới CER cho câu ngắn
        n_syl_est = max(1, len(ref_nfd) // 3)
        cer_thr_eff = cer_threshold + (0.10 if n_syl_est <= 2 else (short_syll_cer_bonus if n_syl_est <= 5 else 0.0))
        gate_by_pair = (cer < cer_thr_eff and mean_neglogp < loss_char_threshold)
        gate_by_nll  = (ref_nll_pc < (ref_nll_threshold + (0.3 if n_syl_est <= 2 else 0.0)))
        gate_by_short = (n_syl_est <= 2 and cer_skel < 0.6)

        # guard len mismatch: tắt short-gate nếu ref dài mà compat token đếm ngắn
        ref_nfd_full = _to_nfd_jamo_only(ref_text_nfd)
        ref_ids_for_gate = _ref_to_vocab_tokens(ref_text_nfd, vocab)
        len_mismatch = (len(ref_nfd_full) >= 4 and len(ref_ids_for_gate) <= 3)
        if len_mismatch:
            gate_by_short = False

        # thêm một nhánh forced dựa mean/max confs — nhưng phanh bởi len_mismatch + CER/Skeleton
        try:
            forced_confs = ctc_char_confidences_from_waveform(
                wav, TARGET_SR, ref_text_nfd,
                blank_vad_thresh=BLANK_VAD_THRESH, vad_pad_ms=VAD_PAD_MS,
                temp=CONF_LOGIT_TEMP, use_confusables=True
            )
            mean_forced = float(np.mean(forced_confs)) if forced_confs else 0.0
            max_forced  = float(np.max(forced_confs))  if forced_confs else 0.0

            _gate_by_forced_raw = (n_syl_est <= 2) and (mean_forced >= FORCED_MEAN_THR) and (max_forced >= FORCED_MAX_THR)
            gate_by_forced = (
                (not len_mismatch) and
                _gate_by_forced_raw and
                ((cer < 0.45) or (cer_skel < 0.45))
            )
        except Exception:
            gate_by_forced = False

        gate_pass = bool(gate_by_pair or gate_by_nll or gate_by_short or gate_by_forced)

        _log.info(f"[gate] mode={_MODE} CER={cer:.3f} CER_skel={cer_skel:.3f} "
                  f"mNLP={mean_neglogp:.3f} NLL/char={ref_nll_pc:.3f} "
                  f"pair={gate_by_pair} nll={gate_by_nll} short={gate_by_short} forced={gate_by_forced} pass={gate_pass}")

        return {
            "cer": float(cer),
            "cer_skeleton": float(cer_skel),
            "mean_neglogp": float(mean_neglogp),
            "ref_nll_per_char": float(ref_nll_pc),
            "gate_by_pair": bool(gate_by_pair),
            "gate_by_nll": bool(gate_by_nll),
            "gate_by_short": bool(gate_by_short),
            "gate_by_forced": bool(gate_by_forced),
            "gate_pass": gate_pass,

            # === Thêm cho HIỂN THỊ ===
            "hyp": hyp,                              # chuỗi decode đã strip (nếu cần)
            "hyp_nfd": hyp_nfd,                      # NFD dùng chấm điểm
            "hyp_display": hyp_display,              # chuỗi thô (có thể còn space/punct)
            "hyp_display_nfd": hyp_display_nfd,      # NFD để HIỂN THỊ (giữ separators)
            "ref_display_nfd": ref_display_nfd,      # NFD của ref để HIỂN THỊ (giữ separators)
        }

def ctc_gate_global(wav_path: str, ref_text_nfd: str, **kwargs) -> Dict[str, float]:
    wav, sr = torchaudio.load(wav_path)
    return ctc_gate_global_from_waveform(wav, sr, ref_text_nfd, **kwargs)
