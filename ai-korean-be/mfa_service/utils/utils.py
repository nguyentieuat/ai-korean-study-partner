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
import unicodedata as ud
import re
from typing import List, Tuple

# các ký tự zero-width hay gặp
_ZW_CODES = [0x200B, 0x200C, 0x200D, 0xFEFF]  # ZWSP, ZWNJ, ZWJ, BOM

def _strip_invisibles(s: str) -> str:
    tbl = {code: None for code in _ZW_CODES}
    # có thể tẩy thêm non-breaking space:
    tbl[0x00A0] = None
    return s.translate(tbl)

def _nfd(s: str) -> str:
    return ud.normalize("NFD", s)

def _is_hangul_syllable(ch: str) -> bool:
    cp = ord(ch)
    return 0xAC00 <= cp <= 0xD7A3

def _is_jamo(ch: str) -> bool:
    cp = ord(ch)
    return 0x1100 <= cp <= 0x11FF

def _syllable_to_jamos_nfd(syl: str) -> List[str]:
    return [c for c in _nfd(syl) if _is_jamo(c)]

# Dấu câu phổ biến để gắn vào NHÃN mảnh cuối cùng
_PUNCT = set(".,!?;:…“”\"'()[]{}-–—~·•、，。；：？！《》〈〉『』「」")
def _first_hangul_index(s: str) -> int:
    for idx, ch in enumerate(s):
        if _is_hangul_syllable(ch):
            return idx
    return -1

def _trim_invis_and_space_left(s: str) -> str:
    # bỏ invisible + khoảng trắng ở đầu
    i = 0
    L = len(s)
    while i < L:
        ch = s[i]
        if ch.isspace() or ord(ch) in _ZW_CODES or ord(ch) == 0x00A0:
            i += 1
        else:
            break
    return s[i:]

def _repair_chunks(chunks: List[Tuple[str, List[str]]]) -> List[Tuple[str, List[str]]]:
    """
    Mỗi mảnh:
      - Loại mọi ký tự đứng TRƯỚC âm tiết Hangul đầu tiên trong label.
      - Phần 'lead' đó sẽ được đẩy vào mảnh trước (nếu có), hoặc bỏ nếu không có.
      - Chỉ giữ đúng 1 âm tiết Hangul ở đầu; phần còn lại (dấu câu sau âm tiết) giữ nguyên.
    """
    fixed: List[Tuple[str, List[str]]] = []
    for label, jamos in chunks:
        if not label:
            continue

        # tỉa invisible/space đầu nhãn trước khi xét
        label = _trim_invis_and_space_left(label)

        hi = _first_hangul_index(label)
        if hi == -1:
            # nhãn không có Hangul → gắn vào mảnh trước (nếu có), bỏ nếu không
            if fixed:
                pl, pj = fixed[-1]
                fixed[-1] = (pl + label, pj)
            continue

        lead = label[:hi]    # rác/ngoặc/space/invisible trước Hangul
        core = label[hi:]    # từ Hangul đầu tiên trở đi

        head = core[0]       # đúng 1 âm tiết Hangul làm đầu nhãn
        tail = core[1:]      # phần sau (dấu câu...), giữ nguyên

        # đẩy phần 'lead' sang mảnh trước (nếu có)
        if lead:
            if fixed:
                pl, pj = fixed[-1]
                fixed[-1] = (pl + lead, pj)
            # nếu không có mảnh trước thì bỏ 'lead'

        fixed.append((head + tail, jamos))
    return fixed
def _sanitize_labels_inplace(chunks):
    for i in range(len(chunks)):
        lab, jamos = chunks[i]
        # cắt toàn bộ ký tự đứng trước âm tiết Hangul đầu tiên
        k = 0
        while k < len(lab) and not _is_hangul_syllable(lab[k]):
            k += 1
        lab = lab[k:] if k < len(lab) else lab
        chunks[i] = (lab, jamos)

def text_to_jamo_chunks(text_ref: str) -> List[Tuple[str, List[str]]]:
    """
    - Duyệt ref: MỖI âm tiết Hangul -> 1 mảnh.
      label = chính âm tiết đó + CHỈ các dấu câu ngay SAU nó (.,!?…)
      jamo_list = jamo của âm tiết TƯƠNG ỨNG trong clean (nếu thiếu thì fallback về ref)
    - Không bao giờ gắn ký tự ĐỨNG TRƯỚC vào nhãn.
    - Bỏ qua khoảng trắng, invisible, và ký tự rác đứng trước.
    """
    # 0) lấy clean (để tách jamo) + tẩy invisible
    try:
        _text_clean, _tags = explain_clean_for_user(text_ref)
    except Exception:
        _text_clean, _tags = text_ref, []

    ref = _strip_invisibles(text_ref)
    cln = _strip_invisibles(_text_clean)

    # 1) gom âm tiết Hangul từ clean (nguồn jamo)
    clean_sylls = [c for c in cln if _is_hangul_syllable(c)]
    k = 0  # con trỏ clean

    chunks: List[Tuple[str, List[str]]] = []
    j = 0
    n = len(ref)

    def _skip_ws_and_invis(idx: int) -> int:
        while idx < n:
            ch = ref[idx]
            if ch.isspace() or ord(ch) in _ZW_CODES or ord(ch) == 0x00A0:
                idx += 1
            else:
                break
        return idx

    while j < n:
        # BỎ MỌI thứ đứng TRƯỚC (space/invisible/rác) — không gắn vào đâu
        j = _skip_ws_and_invis(j)
        if j >= n:
            break

        ch = ref[j]

        if _is_hangul_syllable(ch):
            # 2) tạo mảnh: nhãn = âm tiết hiện tại + dấu câu NGAY SAU nó
            label = ch
            j += 1

            # kéo các dấu câu NGAY SAU (chỉ sau, không bao giờ lấy ký tự trước)
            while j < n and (ref[j] in _PUNCT):
                label += ref[j]
                j += 1

            # 3) jamo từ clean (nếu hết clean thì fallback ref)
            if k < len(clean_sylls):
                ch_c = clean_sylls[k]
                k += 1
            else:
                ch_c = ch
            jamos = _syllable_to_jamos_nfd(ch_c)

            chunks.append((label, jamos))
            continue

        # Không phải Hangul: bỏ qua (không đính vào mảnh sắp tới)
        j += 1

    # Ghi chú nếu  cần dùng:
    notes_ = list(dict.fromkeys(
        n.strip() for n in (e.get("note") for e in _tags)
        if isinstance(n, str) and n.strip()
    ))
    chunks = _repair_chunks(chunks)
    _sanitize_labels_inplace(chunks)
    return chunks, notes_
