# utils/utils_advice.py
import os, json

PRON_RULES_PATH = "./pron_rules_ko.json"

def _load_pron_rules(path=PRON_RULES_PATH):
    try:
        with open(path, "r", encoding="utf-8") as f:
            rules = json.load(f)
    except Exception:
        rules = {}

    # classes ở top-level (giữ kiểu cũ): key -> {covers, advice}
    classes = {k: v for k, v in rules.items() if isinstance(v, dict) and ("covers" in v or "advice" in v)}
    class_covers = {cls: set(cfg.get("covers", [])) for cls, cfg in classes.items()}

    # phoneme -> class
    ph2cls = {}
    for cls, phs in class_covers.items():
        for ph in phs:
            ph2cls.setdefault(ph, cls)

    # xác định class nguyên âm
    vowel_class = None
    for cls in classes.keys():
        if "vowel" in cls.lower():
            vowel_class = cls
            break

    return {
        "raw": rules,
        "classes": classes,
        "class_covers": class_covers,
        "phoneme2class": ph2cls,
        "vowel_class": vowel_class,
        "context_rules": rules.get("context_rules", []),
        "ipa_aliases": rules.get("ipa_aliases", {})
    }

PRON_RULES = _load_pron_rules()

VOWELS = {"a","e","i","o","u","ɐ","ʌ","ɛ","ɯ","ø","y","ɨ","ə","ɤ","ʊ","œ","ɜ","æ","ʉ","ɒ"}
STOP_UNRELEASED = {"k̚","t̚","p̚"}
# Map IPA -> Jamo Hàn để quyết định cặp tương phản áp dụng
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

def _ipa_to_jamo(ph: str) -> str | None:
    ph = _norm_ipa(ph)
    return IPA_TO_JAMO.get(ph)

def _filter_vowel_pair_lines(curr: str, tips: list[str]) -> list[str]:
    """Giữ lại các dòng tip về cặp ㅐ/ㅔ, ㅡ/ㅜ, ㅓ/ㅗ chỉ khi curr thuộc cặp đó."""
    jamo = _ipa_to_jamo(curr)
    out: list[str] = []
    for line in tips:
        matched = None
        for label, members in VOWEL_PAIR_LABELS.items():
            if label in line:
                matched = (label, members)
                break
        if matched is None:
            out.append(line)  # câu tip chung -> luôn giữ
        else:
            _, members = matched
            if jamo and jamo in members:
                out.append(line)  # chỉ giữ nếu curr thuộc cặp
            # ngược lại: bỏ
    return out

def _norm_ipa(ph: str) -> str:
    return PRON_RULES["ipa_aliases"].get(ph, ph)

def _is_vowel(ph: str) -> bool:
    return ph in VOWELS

def _class_of(ph: str) -> str | None:
    ph = _norm_ipa(ph)
    return PRON_RULES["phoneme2class"].get(ph)

def _apply_context_rules(curr: str, prev: str | None, next_: str | None, position: str | None, env_hint: str | None):
    adv = []
    curr_n = _norm_ipa(curr) if curr else ""
    prev_n = _norm_ipa(prev) if prev else ""
    next_n = _norm_ipa(next_) if next_ else ""

    for rule in PRON_RULES["context_rules"]:
        cond = rule.get("if", {})
        ok = True

        # các điều kiện mềm, chỉ check nếu có
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
            # xử lý nhanh L+N / N+L
            if cond["pattern"] == "L+N_or_N+L":
                Lset = {"l","ɾ"}
                Nset = {"n"}
                pair = (curr_n in Lset and next_n in Nset) or (curr_n in Nset and next_n in Lset)
                if not pair:
                    ok = False

        if ok:
            adv.extend(rule.get("advice", []))

    return adv

def advices_for_phoneme(
    curr: str,
    prev: str | None,
    next_: str | None,
    position: str | None,
    low_score: bool = True,
    env_hint: str | None = None,
):
    """
    position: 'onset'|'nucleus'|'coda'|None
    env_hint: gợi ý bối cảnh (vd 'tense_trigger').
    """
    tips: list[str] = []

    # theo class
    cls = _class_of(curr)
    if cls and cls in PRON_RULES["classes"]:
        base = PRON_RULES["classes"][cls].get("advice", []) or []
        # Nếu là lớp nguyên âm -> lọc các dòng có nhắc tới cặp chỉ khi phù hợp
        if PRON_RULES["vowel_class"] and cls == PRON_RULES["vowel_class"]:
            base = _filter_vowel_pair_lines(curr, list(base))
            # Nếu có cấu hình cặp trong JSON, chỉ thêm tip tương ứng cặp hiện tại
            vc = PRON_RULES["classes"][cls]
            jamo = _ipa_to_jamo(curr)
            # 1) cặp viết bằng jamo (như bạn đang dùng)
            for pair in vc.get("pairs", []):
                contrast = set(pair.get("contrast") or [])
                tip = pair.get("tip")
                if tip and jamo and jamo in contrast:
                    base.append(tip)
            # 2) (tuỳ chọn) nếu có cặp theo IPA trong JSON
            for pair in vc.get("pairs_ipa", []):
                contrast_ipa = set(_norm_ipa(x) for x in (pair.get("contrast") or []))
                tip = pair.get("tip")
                if tip and _norm_ipa(curr) in contrast_ipa:
                    base.append(tip)
        tips += base
    elif _is_vowel(_norm_ipa(curr)) and PRON_RULES["vowel_class"] in PRON_RULES["classes"]:
        # curr là nguyên âm nhưng chưa map được class -> dùng lớp nguyên âm mặc định
        base = PRON_RULES["classes"][PRON_RULES["vowel_class"]].get("advice", []) or []
        base = _filter_vowel_pair_lines(curr, list(base))
        tips += base

    # theo ngữ cảnh
    tips += _apply_context_rules(curr, prev, next_, position, env_hint)

    # mẹo bổ sung cho coda stop không nhả hơi
    if curr in STOP_UNRELEASED and "Âm tắc cuối" not in " ".join(tips):
        tips.append("Âm tắc cuối (받침) cần đóng gọn, không nhả hơi; tránh thêm nguyên âm đệm.")

    # rút gọn nếu low_score; nếu không thì vẫn giữ nhưng đã lọc cặp không liên quan
    if low_score and len(tips) > 3:
        tips = tips[:3]
    return tips