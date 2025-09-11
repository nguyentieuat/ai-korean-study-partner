# g2p_needleman_wunsch_alignment/utils.py

def load_mfa_dict(dict_path):
    mfa_dict = {}
    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == "" or line.startswith("#"):
                continue
            parts = line.strip().split()
            key = parts[0]
            phonemes = parts[5:]  # chỉ lấy các phoneme thực sự
            mfa_dict[key] = phonemes
    return mfa_dict

import unicodedata
from typing import Optional, List, Tuple, Dict, Union

# --- Clean text (surface-like spelling) ---
try:
    from ko_phonology import explain_clean_for_user
except Exception:
    # fallback stub nếu module không có (để không vỡ import trong môi trường build)
    def explain_clean_for_user(text: str, opt=None):
        return text, []

_SBASE = 0xAC00
_LBASE = 0x1100  # choseong
_VBASE = 0x1161  # jungseong
_TBASE = 0x11A7  # jongseong (T=0: no coda)
_LCOUNT, _VCOUNT, _TCOUNT = 19, 21, 28
_NCOUNT = _VCOUNT * _TCOUNT

def _is_hangul_syllable(ch: str) -> bool:
    return ch and _SBASE <= ord(ch) <= 0xD7A3

def _decompose_hangul_syllable(syll: str):
    S = ord(syll) - _SBASE
    L = _LBASE + (S // _NCOUNT)
    V = _VBASE + ((S % _NCOUNT) // _TCOUNT)
    T_idx = S % _TCOUNT
    T = (_TBASE + T_idx) if T_idx != 0 else None
    return L, V, T  # codepoints (ints)

# Onset (choseong)
_CHOSEONG_TO_IPA = {
    0x1100:"k", 0x1101:"k͈", 0x1102:"n", 0x1103:"t", 0x1104:"t͈", 0x1105:"ɾ",
    0x1106:"m", 0x1107:"p", 0x1108:"p͈", 0x1109:"s", 0x110A:"s͈", 0x110B:"Ø",
    0x110C:"t͡ɕ", 0x110D:"t͡ɕ͈", 0x110E:"t͡ɕʰ", 0x110F:"kʰ", 0x1110:"tʰ",
    0x1111:"pʰ", 0x1112:"h",
}

# Vowel (jungseong)
_JUNGSEONG_TO_IPA = {
    0x1161:["a"], 0x1162:["ɛ"], 0x1163:["j","a"], 0x1164:["j","ɛ"],
    0x1165:["ʌ"], 0x1166:["e"], 0x1167:["j","ʌ"], 0x1168:["j","e"],
    0x1169:["o"], 0x116A:["w","a"], 0x116B:["w","ɛ"], 0x116C:["w","e"],
    0x116D:["j","o"], 0x116E:["u"], 0x116F:["w","ʌ"], 0x1170:["w","e"],
    0x1171:["w","i"], 0x1172:["j","u"], 0x1173:["ɯ"], 0x1174:["ɯ","i"],
    0x1175:["i"],
}

# Coda (jongseong) — base unreleased
_JONGSEONG_TO_IPA = {
    None: None,
    0x11A8:"k̚", 0x11A9:"k̚", 0x11AA:"k̚",
    0x11AB:"n",  0x11AC:"n",  0x11AD:"n",
    0x11AE:"t̚",
    0x11AF:"l",  0x11B0:"k̚", 0x11B1:"m",  0x11B2:"p̚",
    0x11B3:"t̚", 0x11B4:"t̚", 0x11B5:"p̚", 0x11B6:"l",
    0x11B7:"m",  0x11B8:"p̚", 0x11B9:"p̚",
    0x11BA:"t̚", 0x11BB:"t̚", 0x11BC:"ŋ",  0x11BD:"t̚",
    0x11BE:"t̚", 0x11BF:"k̚", 0x11C0:"t̚", 0x11C1:"p̚", 0x11C2:"t̚",
}

def _emit_onset(L: int, seq: list):
    ipa = _CHOSEONG_TO_IPA.get(L)
    if ipa and ipa != "Ø":  # ㅇ silent onset
        seq.append(ipa)

def _emit_vowel(V: int, seq: list):
    seq.extend(_JUNGSEONG_TO_IPA.get(V, ["?"]))

def _emit_coda(T: Optional[int], seq: list):
    ipa = _JONGSEONG_TO_IPA.get(T)
    if ipa:
        seq.append(ipa)

# Chuẩn hoá một số ký hiệu từ điển về “chuẩn hiển thị”
_SANITIZE = {"ɐ":"a"}
def _sanitize(seq):
    out=[]
    for x in seq:
        out.append(_SANITIZE.get(x, x))
    return out

def _g2p_token(token: str, mfa_dict: dict) -> List[str]:
    """G2P cho một 'token' (đÃ clean). Ưu tiên tra whole-word."""
    res: List[str] = []
    if isinstance(mfa_dict, dict) and token in mfa_dict:
        phs = mfa_dict[token]
        res.extend(_sanitize(phs if isinstance(phs, (list,tuple)) else [phs]))
        return res
    for ch in token:
        if _is_hangul_syllable(ch):
            L,V,T = _decompose_hangul_syllable(ch)
            _emit_onset(L, res)
            _emit_vowel(V, res)
            _emit_coda(T, res)
        else:
            if isinstance(mfa_dict, dict) and ch in mfa_dict:
                phs = mfa_dict[ch]
                res.extend(_sanitize(phs if isinstance(phs,(list,tuple)) else [phs]))
    return res

# --- SMART CHUNKING cho return_by_word=True ---
_CHO_ㅇ = 0x110B
_JONG_ㅅ = 0x11BA
_JONG_ㅆ = 0x11BB

def _chunkify_word_with_s_liaison(word_ref: str, word_clean: str) -> List[Tuple[str, List[str]]]:
    """
    Tách một từ Hangul thành các mảnh (mỗi âm tiết 1 mảnh), nhưng:
      - Nếu âm tiết i có 받침 ㅅ/ㅆ và âm tiết i+1 là nguyên âm (onset ㅇ),
        thì đem ㅅ/ㅆ sang làm onset của mảnh i+1 (→ s / s͈) thay cho coda 't̚'.
      - Nhãn: mặc định là âm tiết gốc (từ word_ref).
        Riêng pattern ... '어' + '요' → nhãn mảnh '어' hiển thị '어요' (phoneme chỉ cho '어').
    Phoneme phát sinh dựa trên *word_clean* (sau normalize_clean) để ổn định.
    """

    # Lấy danh sách âm tiết Hangul (giữ thứ tự)
    syls_ref  = [ch for ch in word_ref  if _is_hangul_syllable(ch)]
    syls_clean= [ch for ch in word_clean if _is_hangul_syllable(ch)]
    if not syls_clean:
        # không phải hangul → trả 1 mảnh đơn giản
        return [(word_ref, _g2p_token(word_clean, {}))]

    # Nếu độ dài lệch, cố gắng dùng min; còn lại gộp vào mảnh cuối
    n = min(len(syls_ref), len(syls_clean))
    chunks: List[Tuple[str, List[str]]] = []
    carry_next: List[str] = []

    for i in range(n):
        sr = syls_ref[i]
        sc = syls_clean[i]
        Lc, Vc, Tc = _decompose_hangul_syllable(sc)  # dùng clean cho phoneme
        # Nhãn mặc định theo ref
        label = sr

        phones: List[str] = []
        # prepend carry (s hoặc s͈) nếu có
        if carry_next:
            phones.extend(carry_next)
            carry_next = []

        # onset (bỏ ㅇ)
        _emit_onset(Lc, phones)
        # vowel
        _emit_vowel(Vc, phones)

        # quyết định coda / carry sang sau
        # cần dựa vào *ref* (vì liaison ㅅ/ㅆ là hiện tượng hình thái/âm vị chữ viết)
        Lr, Vr, Tr = _decompose_hangul_syllable(sr)
        next_is_vowel = False
        if (i+1) < n:
            Ln, Vn, Tn = _decompose_hangul_syllable(syls_clean[i+1])
            next_is_vowel = (Ln == _CHO_ㅇ)

        if next_is_vowel and Tr in (_JONG_ㅅ, _JONG_ㅆ):
            # KHÔNG phát coda t̚; thay vào đó chuẩn bị carry onset cho mảnh sau
            carry_next = ["s͈"] if Tr == _JONG_ㅆ else ["s"]
        else:
            # phát coda bình thường (nếu có)
            _emit_coda(Tc, phones)

        chunks.append((label, _sanitize(phones)))

    # Nếu còn dư âm tiết (do ref dài hơn clean), gộp vào mảnh cuối
    if len(syls_ref) > n:
        rest_label = "".join(syls_ref[n:])
        if chunks:
            last_label, last_phs = chunks[-1]
            chunks[-1] = (last_label + rest_label, last_phs)
        else:
            chunks.append((rest_label, []))

    # Nếu trong word còn non-Hangul (dấu câu...), gắn vào mảnh cuối để giữ nhãn
    tail_non_h = "".join([ch for ch in word_ref if not _is_hangul_syllable(ch)])
    if tail_non_h and chunks:
        last_label, last_phs = chunks[-1]
        chunks[-1] = (last_label + tail_non_h, last_phs)

    return chunks

def text_to_phonemes_mfa(
    text_ref: str,
    mfa_dict: dict,
    *,
    return_by_word: bool = False
) -> Union[List[str], List[Tuple[str, List[str]]]]:
    """
    G2P với hỗ trợ clean text & trả về theo mảnh trong từ (smart) khi return_by_word=True.

    - Luôn clean trước: normalize_clean(text_ref) -> text_clean
    - return_by_word=False (mặc định): trả List[str] (phoneme phẳng) như cũ.
    - return_by_word=True: trả List[ (label_mảnh, [phonemes]) ].
      *Mảnh* thường là mỗi âm tiết Hangul

    Ghi chú:
    - Nếu từ có non-Hangul, mảnh cuối sẽ được ghép phần non-H vào nhãn để giữ nguyên hiển thị.
    """
    # 0) Clean text
    try:
        text_clean, _tags = explain_clean_for_user(text_ref)
    except Exception:
        text_clean = text_ref

    if not return_by_word:
        # Phẳng: ưu tiên tra theo từ trong dict, fallback phân rã âm tiết
        res: List[str] = []
        for word in text_clean.strip().split():
            res.extend(_g2p_token(word, mfa_dict))
        return res

    # by "word" (nhưng tách smart theo âm tiết bên trong)
    out: List[Tuple[str, List[str]]] = []
    ref_tokens   = text_ref.strip().split()
    clean_tokens = text_clean.strip().split()
    n = max(len(ref_tokens), len(clean_tokens))
    if len(ref_tokens)   < n: ref_tokens   += [""]*(n-len(ref_tokens))
    if len(clean_tokens) < n: clean_tokens += [""]*(n-len(clean_tokens))

    for w_ref, w_clean in zip(ref_tokens, clean_tokens):
        # Nếu toàn Hangul → tách smart theo âm tiết + ㅅ/ㅆ liaison
        if all(_is_hangul_syllable(ch) for ch in w_ref) and all(_is_hangul_syllable(ch) for ch in w_clean):
            chunks = _chunkify_word_with_s_liaison(w_ref, w_clean)
            # Nếu từ có trong từ điển MFA nguyên từ, có thể cân nhắc override phones theo dict.
            # Ở đây ta giữ phones theo tách-âm-tiết để bám sát ví dụ mong muốn.
            out.extend(chunks)
        else:
            # Không phải thuần Hangul → trả một mảnh theo cả từ
            out.append((w_ref, _g2p_token(w_clean, mfa_dict)))
    
    notes_ = list(dict.fromkeys(
                n.strip() for n in (e.get("note") for e in _tags)
                if isinstance(n, str) and n.strip()
            ))
    return out, notes_
