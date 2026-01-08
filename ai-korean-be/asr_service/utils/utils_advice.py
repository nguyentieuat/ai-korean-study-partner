# utils/utils_advice.py
# -----------------------------------------------------------------------------
# Trả về "mẹo" (advice) phát âm theo từng Jamo (NFD) hoặc IPA.
# - Đọc luật từ JSON (pron_rules_ko.json).
# - Hỗ trợ nucleus kép (ᅭ/ᅲ/ᅧ/ᅪ/ᅫ/ᅯ/ᅬ/ᅱ/ᅴ): tách glide + vowel, ưu tiên mẹo cho nguyên âm lõi.
# - Sửa lỗi: nguyên âm đơn (ví dụ ᅥ=ʌ) vẫn nhận được tip ngay cả khi không có class mapping cụ thể,
#   bằng cách (a) phát hiện vowel_class thông minh hơn, (b) fallback về tip chung của vowel_class.
# -----------------------------------------------------------------------------

from typing import Optional, List, Dict
import os
import json
import unicodedata as ud

PRON_RULES_PATH = "./pron_rules_ko.json"

# =======================
# Load & giữ cấu hình luật
# =======================
def _load_pron_rules(path: str = PRON_RULES_PATH) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            rules = json.load(f)
    except Exception:
        rules = {}

    classes = {
        k: v for k, v in rules.items()
        if isinstance(v, dict) and ("covers" in v or "advice" in v)
    }
    class_covers = {cls: set(cfg.get("covers", [])) for cls, cfg in classes.items()}

    ph2cls: Dict[str, str] = {}
    for cls, phs in class_covers.items():
        for ph in phs:
            ph2cls.setdefault(ph, cls)

    # 1) ưu tiên tìm class có "vowel" trong tên
    vowel_class = None
    for cls in classes.keys():
        if "vowel" in cls.lower():
            vowel_class = cls
            break

    # 2) nếu vẫn chưa có, suy luận class nguyên âm theo covers giao với VOWELS
    if vowel_class is None:
        best_cls = None
        best_overlap = -1
        for cls, covers in class_covers.items():
            overlap = len({_norm_ipa(x) for x in covers} & VOWELS)
            if overlap > best_overlap:
                best_overlap = overlap
                best_cls = cls
        # chấp nhận nếu covers giao >= 3 nguyên âm (tránh nhầm)
        if best_overlap >= 3:
            vowel_class = best_cls

    return {
        "raw": rules,
        "classes": classes,
        "class_covers": class_covers,
        "phoneme2class": ph2cls,
        "vowel_class": vowel_class,
        "context_rules": rules.get("context_rules", []),
        "ipa_aliases": rules.get("ipa_aliases", {})
    }

# Tải một lần
PRON_RULES = _load_pron_rules()

# =======================
# Hằng & bảng tra cứu
# =======================
VOWELS = {
    "a", "e", "i", "o", "u",
    "ɐ", "ʌ", "ɛ", "ɯ", "ø", "y", "ɨ", "ə", "ɤ",
    "ʊ", "œ", "ɜ", "æ", "ʉ", "ɒ"
}
STOP_UNRELEASED = {"k̚", "t̚", "p̚"}

# Map IPA -> Jamo (để lọc mẹo theo cặp tương phản ㅐ/ㅔ, ㅡ/ㅜ, ㅓ/ㅗ)
IPA_TO_JAMO = {
    "ɛ": "ㅐ",
    "e": "ㅔ",  "e̞": "ㅔ",
    "ɯ": "ㅡ",
    "u": "ㅜ",
    "ʌ": "ㅓ",
    "o": "ㅗ",  "o̞": "ㅗ",
}
VOWEL_PAIR_LABELS = {
    "ㅐ/ㅔ": {"ㅐ", "ㅔ"},
    "ㅡ/ㅜ": {"ㅡ", "ㅜ"},
    "ㅓ/ㅗ": {"ㅓ", "ㅗ"},
}

# --- Jamo → IPA (thô) theo vị trí ---
JAMO_ONSET_TO_IPA = {
    "ᄀ": "k", "ᄁ": "k", "ᄂ": "n", "ᄃ": "t", "ᄄ": "t", "ᄅ": "ɾ",
    "ᄆ": "m", "ᄇ": "p", "ᄈ": "p", "ᄉ": "s", "ᄊ": "s",
    "ᄋ": None,  # onset ᄋ câm
    "ᄌ": "tɕ", "ᄍ": "tɕ", "ᄎ": "tɕʰ",
    "ᄏ": "kʰ", "ᄐ": "tʰ", "ᄑ": "pʰ", "ᄒ": "h",
}
JAMO_NUCLEUS_TO_IPA = {
    "ᅡ": "a",  "ᅢ": "ɛ",
    "ᅣ": "ja", "ᅤ": "jɛ",
    "ᅥ": "ʌ",  "ᅦ": "e",
    "ᅧ": "jʌ", "ᅨ": "je",
    "ᅩ": "o",  "ᅪ": "wa", "ᅫ": "wɛ", "ᅬ": "we",
    "ᅭ": "jo",
    "ᅮ": "u",  "ᅯ": "wʌ", "ᅰ": "we", "ᅱ": "wi",
    "ᅲ": "ju",
    "ᅳ": "ɯ",
    "ᅴ": "ɯi",   # chuẩn hoá ɰi -> ɯi ở _norm_ipa/_split_nucleus_ipa
    "ᅵ": "i",
}
JAMO_CODA_TO_IPA = {
    "ᆨ": "k̚", "ᆩ": "k̚", "ᆪ": "k̚",
    "ᆫ": "n", "ᆬ": "n", "ᆭ": "n",
    "ᆮ": "t̚",
    "ᆯ": "l",
    "ᆰ": "k̚", "ᆱ": "m", "ᆲ": "p̚", "ᆳ": "t̚", "ᆴ": "t̚", "ᆵ": "p̚", "ᆶ": "t̚",
    "ᆷ": "m",
    "ᆸ": "p̚", "ᆹ": "p̚",
    "ᆺ": "t̚", "ᆻ": "t̚",
    "ᆼ": "ŋ",
    "ᆽ": "t̚", "ᆾ": "t̚", "ᆿ": "k̚", "ᇀ": "t̚", "ᇁ": "p̚", "ᇂ": "t̚",
}

# =======================
# Tiện ích Jamo / Hangul
# =======================
def _is_jamo(ch: str) -> bool:
    if not ch:
        return False
    cp = ord(ch)
    return 0x1100 <= cp <= 0x11FF

def _is_hangul_syllable(ch: str) -> bool:
    if not ch:
        return False
    cp = ord(ch)
    return 0xAC00 <= cp <= 0xD7A3

def _decompose_to_jamo(syl: str) -> List[str]:
    nfd = ud.normalize("NFD", syl)
    return [c for c in nfd if _is_jamo(c)]

# =======================
# Chuẩn hoá & tách IPA
# =======================
def _norm_ipa(ph: str) -> str:
    if not ph:
        return ph
    ph = PRON_RULES.get("ipa_aliases", {}).get(ph, ph)
    if ph == "o̞":
        ph = "o"
    if ph == "e̞":
        ph = "e"
    if ph == "ɰ":
        ph = "ɯ"
    return ph

def _split_nucleus_ipa(ipa: Optional[str]) -> List[str]:
    """
    Trả về [glide?, vowel core]; ưu tiên phần tử cuối là nguyên âm lõi.
    Ví dụ: "jo"->["j","o"], "wʌ"->["w","ʌ"], "ɯi"->["ɯ","i"], "i"->["i"]
    """
    if not ipa:
        return []
    s = _norm_ipa(ipa)
    if s in ("ɰi", "ɯi", "ɰɪ", "ɯɪ"):
        return ["ɯ", "i"]
    if len(s) >= 2 and s[0] in ("j", "w"):
        return [s[0], _norm_ipa(s[1:])]
    return [s]

# =======================
# Map IPA ↔ Jamo cho mẹo cặp
# =======================
def _ipa_to_jamo(ph: str) -> Optional[str]:
    ph = _norm_ipa(ph)
    return IPA_TO_JAMO.get(ph)

def _filter_vowel_pair_lines(curr: str, tips: List[str]) -> List[str]:
    """Giữ các dòng tip cặp ㅐ/ㅔ, ㅡ/ㅜ, ㅓ/ㅗ nếu nguyên âm hiện tại thuộc cặp đó."""
    jamo = _ipa_to_jamo(curr)
    out: List[str] = []
    for line in tips:
        matched = None
        for label, members in VOWEL_PAIR_LABELS.items():
            if label in line:
                matched = (label, members)
                break
        if matched is None:
            out.append(line)
        else:
            _, members = matched
            if jamo and jamo in members:
                out.append(line)
    return out

def _is_vowel(ph: Optional[str]) -> bool:
    return (ph in VOWELS) if ph else False

def _class_of(ph: Optional[str]) -> Optional[str]:
    if not ph:
        return None
    ph = _norm_ipa(ph)
    return PRON_RULES["phoneme2class"].get(ph)

# =======================
# Áp dụng rules ngữ cảnh
# =======================
def _apply_context_rules(
    curr: Optional[str], prev: Optional[str], next_: Optional[str],
    position: Optional[str], env_hint: Optional[str],
) -> List[str]:
    adv: List[str] = []
    curr_n = _norm_ipa(curr) if curr else ""
    prev_n = _norm_ipa(prev) if prev else ""
    next_n = _norm_ipa(next_) if next_ else ""

    for rule in PRON_RULES["context_rules"]:
        cond = rule.get("if", {})
        ok = True
        if "current_in" in cond and curr_n not in cond["current_in"]:
            ok = False
        if ok and "next_in" in cond and next_n not in cond["next_in"]:
            ok = False
        if ok and "prev_in" in cond and prev_n not in cond["prev_in"]:
            ok = False
        if ok and "current_class_in" in cond:
            c = _class_of(curr_n)
            if c not in cond["current_class_in"]:
                ok = False
        if ok and cond.get("next_is_vowel") is True and not _is_vowel(next_n):
            ok = False
        if ok and "position" in cond and position != cond["position"]:
            ok = False
        if ok and "env_hint" in cond and cond["env_hint"] != env_hint:
            ok = False
        if ok and "pattern" in cond:
            if cond["pattern"] == "L+N_or_N+L":
                Lset = {"l", "ɾ"}
                Nset = {"n"}
                pair = (curr_n in Lset and next_n in Nset) or (curr_n in Nset and next_n in Lset)
                if not pair:
                    ok = False
        if ok:
            adv.extend(rule.get("advice", []))
    return adv

# =======================
# Entry: advice theo PHONEME (IPA)
# =======================
def _vowel_base_tips(ipa_vowel: str) -> List[str]:
    """
    Lấy tip chung cho nguyên âm từ vowel_class (nếu có).
    Dùng khi không có mapping cụ thể (class_of(curr) = None) nhưng vẫn là nguyên âm.
    """
    tips: List[str] = []
    vcls = PRON_RULES.get("vowel_class")
    classes = PRON_RULES.get("classes", {})
    if vcls and vcls in classes:
        base = classes[vcls].get("advice", []) or []
        base = _filter_vowel_pair_lines(ipa_vowel, list(base))
        tips += base
    return tips

def advices_for_phoneme(
    curr: Optional[str], prev: Optional[str], next_: Optional[str],
    position: Optional[str], low_score: bool = True, env_hint: Optional[str] = None,
) -> List[str]:
    tips: List[str] = []

    # 1) Theo class
    cls = _class_of(curr)
    if cls and cls in PRON_RULES["classes"]:
        base = PRON_RULES["classes"][cls].get("advice", []) or []
        if PRON_RULES["vowel_class"] and cls == PRON_RULES["vowel_class"]:
            base = _filter_vowel_pair_lines(curr or "", list(base))
            vc = PRON_RULES["classes"][cls]
            jamo = _ipa_to_jamo(curr or "")
            for pair in vc.get("pairs", []):
                contrast = set(pair.get("contrast") or [])
                tip = pair.get("tip")
                if tip and jamo and jamo in contrast:
                    base.append(tip)
            for pair in vc.get("pairs_ipa", []):
                contrast_ipa = set(_norm_ipa(x) for x in (pair.get("contrast") or []))
                tip = pair.get("tip")
                if tip and _norm_ipa(curr or "") in contrast_ipa:
                    base.append(tip)
        tips += base

    # 2) Theo ngữ cảnh
    tips += _apply_context_rules(curr, prev, next_, position, env_hint)

    # 3) Bổ sung mẹo cho coda tắc không nhả hơi
    if curr in STOP_UNRELEASED and "Âm tắc cuối" not in " ".join(tips):
        tips.append("Âm tắc cuối (받침) cần đóng gọn, không nhả hơi; tránh thêm nguyên âm đệm.")

    # 4) Fallback: nếu là nguyên âm nhưng chưa có tip từ class/context → lấy tip chung của vowel_class
    if not tips and _is_vowel(_norm_ipa(curr or "")):
        tips += _vowel_base_tips(curr or "")

    if low_score and len(tips) > 3:
        tips = tips[:3]
    return tips

# =======================
# Entry: advice theo JAMO (NFD)
# =======================
def _jamo_to_ipa(j: Optional[str], position: Optional[str]) -> Optional[str]:
    if not j or not position:
        return None
    if position == "onset":
        return JAMO_ONSET_TO_IPA.get(j)
    if position == "nucleus":
        return JAMO_NUCLEUS_TO_IPA.get(j)
    if position == "coda":
        return JAMO_CODA_TO_IPA.get(j)
    return None

def advices_for_jamo(
    curr_jamo: Optional[str], prev_jamo: Optional[str], next_jamo: Optional[str],
    position: Optional[str], low_score: bool = True, env_hint: Optional[str] = None,
) -> List[str]:
    """
    Wrapper cho Jamo NFD:
    - NFC → tách Jamo đúng vị trí.
    - Jamo → IPA thô theo vị trí.
    - Nucleus kép: tách glide + vowel, ưu tiên lấy mẹo theo **core vowel**; thêm tip ngắn cho glide.
    - Nucleus đơn (ví dụ ᅥ=ʌ): nếu không có mapping class, vẫn trả tip chung của vowel_class.
    """
    def as_jamo(x: Optional[str]) -> Optional[str]:
        if not x:
            return None
        if _is_hangul_syllable(x):
            parts = _decompose_to_jamo(x)
            if position == "onset":
                return parts[0] if len(parts) >= 1 else None
            if position == "nucleus":
                return parts[1] if len(parts) >= 2 else None
            if position == "coda":
                return parts[2] if len(parts) >= 3 else None
            return None
        return x if _is_jamo(x) else None

    cj = as_jamo(curr_jamo)
    pj = as_jamo(prev_jamo) if prev_jamo else None
    nj = as_jamo(next_jamo) if next_jamo else None

    curr_ipa = _jamo_to_ipa(cj, position) if cj else None
    prev_ipa = _jamo_to_ipa(
        pj,
        "onset" if position == "nucleus" else "nucleus" if position == "coda" else "onset"
    ) if pj else None
    next_ipa = _jamo_to_ipa(
        nj,
        "nucleus" if position == "onset" else "coda" if position == "nucleus" else "coda"
    ) if nj else None

    # Onset ᄋ: câm → vẫn thử context (thường rỗng)
    if position != "nucleus" or not curr_ipa:
        return advices_for_phoneme(curr_ipa, prev_ipa, next_ipa, position, low_score=low_score, env_hint=env_hint)

    # Nucleus: tách diphthong
    comps = _split_nucleus_ipa(curr_ipa)
    tips: List[str] = []

    if comps:
        core_vowel = comps[-1]
        tips += advices_for_phoneme(core_vowel, prev_ipa, next_ipa, position, low_score=low_score, env_hint=env_hint)

        # Fallback thêm lần nữa nếu vẫn rỗng (core là nguyên âm đơn nhưng chưa có tip)
        if not tips and _is_vowel(_norm_ipa(core_vowel)):
            tips += _vowel_base_tips(core_vowel)

    # Tip ngắn cho glide
    if low_score and len(comps) == 2 and comps[0] in ("j", "w"):
        if comps[0] == "j":
            tips.append("Glide **j-** (y-) dẫn nhanh vào nguyên âm; đừng tách thành âm tiết riêng.")
        else:  # w
            tips.append("Glide **w-**: tròn môi nhanh rồi chuyển vào nguyên âm; tránh kéo dài thành 'u'.")

    if low_score and len(tips) > 3:
        tips = tips[:3]
    return tips
