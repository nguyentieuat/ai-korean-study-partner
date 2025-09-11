# symbols.py — Jamo pipeline (compat Jamo + ▁ + ·)
# - '_' là PAD
# - ' ' (space) dùng ngăn Jamo trong một âm tiết
# - '▁' là ranh giới từ (word boundary)
# - '·' là ranh giới âm tiết (syllable boundary)
# - SPACE_ID = index('▁')

_pad = "_"

# separators: thứ tự đặt sớm để cố định chỉ số
_separators = "▁· "  # NOTE: phải có cả ' ' và '▁' và '·'

# dấu câu thường gặp (giữ gọn)
_punctuation = ".,!?;:–—-…\"'()[]{}“”’"

# Latin (phòng trường hợp có từ vay mượn như 'meeting')
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# digits (thực tế cleaner đã chuẩn hoá số → Hangeul, nên phần này dự phòng)
_digits = "0123456789"

# --- Korean Compatibility Jamo (U+3131..U+318E) ---
# choseong / jungseong / jongseong theo compat
_jamo_initial = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_jamo_medial  = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_jamo_final   = "ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"

# -------- build symbols (không trùng) --------
symbols = []
_seen = set()

def _push(seq: str):
    for ch in seq:
        if ch not in _seen:
            _seen.add(ch)
            symbols.append(ch)

_push(_pad)
_push(_separators)
_push(_punctuation)
_push(_letters)
_push(_digits)
_push(_jamo_initial + _jamo_medial + _jamo_final)

# ranh giới từ là '▁' (khớp với korean_cleaners(space_token='▁'))
SPACE_ID = symbols.index("▁")

__all__ = ["symbols", "SPACE_ID"]
