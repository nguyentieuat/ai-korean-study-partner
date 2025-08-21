import whisper
from textgrid import TextGrid
import numpy as np
import torch
from .utils import load_mfa_dict, text_to_phonemes_mfa

# --------- 1. Load ASR model once ---------
use_gpu = torch.cuda.is_available()
model = whisper.load_model("small", device="cuda" if use_gpu else "cpu")
if use_gpu:
    model = model.half()

MFA_DICT_PATH = "./korean_mfa.dict"

# ====== Bộ quy tắc tương đương (điểm mềm) ======
# Cho phép một số cặp coi là đúng (1.0) hoặc gần đúng (0.7~0.9)
PHONEME_EQUIVALENCE = {
    ("ㄱ", "ㅋ"): 1.0, ("ㅋ", "ㄱ"): 1.0,
    ("ㄷ", "ㅌ"): 1.0, ("ㅌ", "ㄷ"): 1.0,
    ("ㅂ", "ㅍ"): 1.0, ("ㅍ", "ㅂ"): 1.0,

    ("ㄷ", "ㅈ"): 0.85, ("ㅈ", "ㄷ"): 0.85,  # ㄷ + 이 → 지
    ("ㄷ", "ㅊ"): 0.85, ("ㅊ", "ㄷ"): 0.85,  # ㄷ + 이 → 치
    ("ㅂ", "ㅁ"): 0.9,  ("ㅁ", "ㅂ"): 0.9,   # ㅂ + ㄴ/ㅁ → ㅁ
    ("ㄱ", "ㅇ"): 0.85, ("ㅇ", "ㄱ"): 0.85,  # ㄱ + ㄴ/ㅁ → ㅇ (đồng hoá)

    ("ㄱ", "ㄲ"): 0.9,  ("ㄲ", "ㄱ"): 0.9,   # bật mạnh
    ("ㄷ", "ㄸ"): 0.9,  ("ㄸ", "ㄷ"): 0.9,
    ("ㅂ", "ㅃ"): 0.9,  ("ㅃ", "ㅂ"): 0.9,
    ("ㅈ", "ㅉ"): 0.9,  ("ㅉ", "ㅈ"): 0.9,
}

# ====== Quy tắc biến đổi cặp liên tiếp (áp dụng trước khi align) ======
# Mục tiêu: normalize cả ref & user thành những dạng thường gặp trong phát âm tự nhiên
ASSIMILATION_RULES = [
    # ㄷ + 이 → ㅈ/ㅊ + 이
    (["ㄷ", "이"], ["ㅈ", "이"]),
    (["ㄷ", "이"], ["ㅊ", "이"]),

    # ㄱ + ㄹ → ㅇ + ㄴ (국립 → 궁닙)
    (["ㄱ", "ㄹ"], ["ㅇ", "ㄴ"]),

    # ㄴ + ㄹ → ㄹ + ㄹ (신라 → 실라)
    (["ㄴ", "ㄹ"], ["ㄹ", "ㄹ"]),

    # ㅂ + ㄴ → ㅁ + ㄴ
    (["ㅂ", "ㄴ"], ["ㅁ", "ㄴ"]),
]

def apply_assimilation(seq):
    """
    Áp dụng một số quy tắc biến âm phổ biến theo cặp liên tiếp (greedy, một lượt).
    Không phá vỡ độ dài quá mức; chỉ thay thế cặp -> cặp.
    """
    if not seq:
        return seq
    s = seq[:]
    i = 0
    while i < len(s) - 1:
        applied = False
        window = s[i:i+2]
        for src, tgt in ASSIMILATION_RULES:
            if window == src:
                s[i:i+2] = tgt
                applied = True
                break
        i += 1 if not applied else 2  # nếu có thay thế, nhảy qua cặp vừa thay
    return s

# ====== Quy tắc nối âm/đồng hoá theo ngữ cảnh (để so khớp mềm theo từng vị trí) ======
def get_allowed_phonemes(current_ph, next_ph):
    """
    Trả về tập phoneme chấp nhận được cho current_ph dựa vào next_ph (ngữ cảnh phải-trước).
    Dùng cho so khớp mềm ở bước chấm điểm sau khi align.
    """
    allowed = {current_ph}  # luôn cho phép phoneme gốc

    # ---- 연음 (nối âm): 받침 + nguyên âm đầu → âm cuối nối sang đầu ----
    # Với cấp phoneme rời, ta khó resyllabify hoàn chỉnh; ở đây ta chỉ nới lỏng so khớp:
    # ㄱ/ㄲ/ㅋ + (ㅇ|이|ㅣ) → chấp nhận ㄱ
    if current_ph in ["ㄱ", "ㄲ", "ㅋ"] and next_ph in ["ㅇ", "이", "ㅣ"]:
        allowed.add("ㄱ")

    # ㄷ/ㅌ + (ㅇ|이|ㅣ) → cho phép ㅈ/ㅊ (같이 → 가치/가치)
    if current_ph in ["ㄷ", "ㅌ"] and next_ph in ["ㅇ", "이", "ㅣ"]:
        allowed.update(["ㅈ", "ㅊ"])

    # ㅂ/ㅍ + (ㅇ|이|ㅣ) → cho phép ㅂ (ổn định)
    if current_ph in ["ㅂ", "ㅍ"] and next_ph in ["ㅇ", "이", "ㅣ"]:
        allowed.add("ㅂ")

    # ---- 받침 동화 (đồng hoá phụ âm cuối trước ㄴ/ㅁ) ----
    if current_ph == "ㅂ" and next_ph in ["ㅁ", "ㄴ"]:
        allowed.add("ㅁ")
    if current_ph == "ㄱ" and next_ph in ["ㅁ", "ㄴ"]:
        allowed.add("ㅇ")
    if current_ph == "ㄷ" and next_ph in ["ㅁ", "ㄴ"]:
        allowed.add("ㄴ")

    # ---- ㅎ biến âm/triệt tiêu ----
    if current_ph == "ㅎ":
        allowed.add("")  # có thể rơi
    if current_ph == "ㅎ" and next_ph in ["ㄱ", "ㄷ", "ㅈ"]:
        mapping = {"ㄱ": "ㅋ", "ㄷ": "ㅌ", "ㅈ": "ㅊ"}
        allowed.add(mapping[next_ph])

    # ---- Căng hoá khi gặp phụ âm giống nhau ----
    if current_ph == "ㄱ" and next_ph == "ㄱ":
        allowed.add("ㄲ")
    if current_ph == "ㄷ" and next_ph == "ㄷ":
        allowed.add("ㄸ")
    if current_ph == "ㅂ" and next_ph == "ㅂ":
        allowed.add("ㅃ")
    if current_ph == "ㅈ" and next_ph == "ㅈ":
        allowed.add("ㅉ")

    return allowed

# ====== So khớp xấp xỉ 2 phoneme (không cần đúng tuyệt đối) ======
def compare_phonemes(u_ph, r_ph, r_next=None):
    """
    Trả về điểm 0..1 cho cặp phoneme user vs reference, có xét tương đương & ngữ cảnh.
    r_next: phoneme tham chiếu kế tiếp (để xét nối âm/đồng hoá).
    """
    if u_ph == r_ph:
        return 1.0

    # Cho phép r_ph rơi (vd ㅎ) nếu luật cho phép và user phát âm trống ("")
    allowed = get_allowed_phonemes(r_ph, r_next or "")
    if u_ph in allowed:
        return 1.0

    # Nếu không thuộc allowed theo ngữ cảnh, thử equivalence “gần đúng”
    if (u_ph, r_ph) in PHONEME_EQUIVALENCE:
        return PHONEME_EQUIVALENCE[(u_ph, r_ph)]
    if (r_ph, u_ph) in PHONEME_EQUIVALENCE:
        return PHONEME_EQUIVALENCE[(r_ph, u_ph)]

    return 0.0

# ====== Needleman–Wunsch (giữ nguyên) ======
def needleman_wunsch(seq1, seq2, match=1, mismatch=0, gap=-1):
    try:
        n, m = len(seq1)+1, len(seq2)+1
        score = np.zeros((n, m))
        for i in range(n): score[i, 0] = i * gap
        for j in range(m): score[0, j] = j * gap
        for i in range(1, n):
            for j in range(1, m):
                diag = score[i-1, j-1] + (match if seq1[i-1] == seq2[j-1] else mismatch)
                score[i, j] = max(diag, score[i-1, j] + gap, score[i, j-1] + gap)
        aligned1, aligned2 = [], []
        i, j = n-1, m-1
        while i > 0 or j > 0:
            if i>0 and j>0 and score[i, j] == score[i-1, j-1] + (match if seq1[i-1]==seq2[j-1] else mismatch):
                aligned1.insert(0, seq1[i-1]); aligned2.insert(0, seq2[j-1]); i -= 1; j -= 1
            elif i>0 and score[i, j] == score[i-1, j] + gap:
                aligned1.insert(0, seq1[i-1]); aligned2.insert(0, '-'); i -= 1
            else:
                aligned1.insert(0, '-'); aligned2.insert(0, seq2[j-1]); j -= 1
        return aligned1, aligned2
    except Exception as e:
        print("[ERROR] Needleman-Wunsch alignment failed:", e)
        return [], []

def evaluate_chunk(chunk_wave: np.ndarray, reference_text: str, textgrid_file: str):
    try:
        # 1) Chạy ASR cho chunk
        result = model.transcribe(chunk_wave, language='ko', fp16=use_gpu)
        chunk_text = result.get("text", "").strip()

        # 2) Load MFA dict & chuyển text → phoneme
        try:
            mfa_dict = load_mfa_dict(MFA_DICT_PATH)
        except Exception as e:
            print("[ERROR] Loading MFA dict failed:", e)
            mfa_dict = {}

        ref_phonemes = text_to_phonemes_mfa(reference_text, mfa_dict) or []
        user_phonemes = text_to_phonemes_mfa(chunk_text, mfa_dict) or []

        # 3) Đọc TextGrid (phones tier)
        try:
            tg = TextGrid()
            tg.read(textgrid_file)
            ref_tier = tg.getFirst("phones")
            ref_phonemes_with_time = [
                (p.mark, p.minTime, p.maxTime)
                for p in ref_tier if p.mark.strip() != ""
            ]
            tg_phonemes_only = [p[0] for p in ref_phonemes_with_time]
        except Exception as e:
            print("[ERROR] Reading TextGrid failed:", e)
            ref_phonemes_with_time = []
            tg_phonemes_only = []

        # 4) Chuẩn hoá (normalize) theo các quy tắc biến âm phổ biến (áp dụng cho cả 2)
        ref_norm = apply_assimilation(tg_phonemes_only if tg_phonemes_only else ref_phonemes)
        user_norm = apply_assimilation(user_phonemes)

        # 5) Align user_norm ↔ ref_norm
        aligned_user, aligned_ref = needleman_wunsch(user_norm, ref_norm)

        # 6) Tính điểm từng phoneme với so khớp mềm theo ngữ cảnh
        phoneme_scores = []
        ref_index = 0  # index chạy trên ref_phonemes_with_time (để lấy time)
        for k, (u, r) in enumerate(zip(aligned_user, aligned_ref)):
            if r == '-':
                continue
            # Tìm next_ref (bỏ qua gap)
            r_next = None
            for t in range(k+1, len(aligned_ref)):
                if aligned_ref[t] != '-':
                    r_next = aligned_ref[t]
                    break

            # Tính điểm mềm
            s = compare_phonemes(u if u != '-' else "", r, r_next)

            # Lấy time từ TextGrid nếu có
            if ref_index < len(ref_phonemes_with_time):
                r_label, start, end = ref_phonemes_with_time[ref_index]
            else:
                r_label, start, end = r, None, None

            feedback = "tốt" if s >= 0.99 else f"nên chỉnh '{r_label}' (u='{u}')"
            phoneme_scores.append({
                "phoneme": r_label,
                "start": start, "end": end,
                "score": float(s),
                "feedback": feedback
            })
            ref_index += 1

        # 7) Điểm trung bình 0..1
        avg_score = 0.0
        if phoneme_scores:
            avg_score = float(sum(p["score"] for p in phoneme_scores) / len(phoneme_scores))

        # 8) Gom theo từ (fallback đơn giản nếu không có mapping time->word)
        words = reference_text.split()
        word_scores = []
        if words and phoneme_scores:
            # phân bổ đều số phoneme cho mỗi từ (giả định)
            ppw = max(1, len(phoneme_scores) // len(words))
            for i, w in enumerate(words):
                start_idx = i * ppw
                end_idx = min((i+1)*ppw, len(phoneme_scores)) - 1
                if start_idx > end_idx:
                    # hết phoneme
                    word_scores.append({"word": w, "start": None, "end": None, "correct": False, "score": 0.0})
                    continue
                seg = phoneme_scores[start_idx:end_idx+1]
                # điểm từ = trung bình phoneme trong segment
                w_score = float(sum(p["score"] for p in seg) / len(seg))
                word_scores.append({
                    "word": w,
                    "start": seg[0]["start"],
                    "end": seg[-1]["end"],
                    "correct": w_score >= 0.8,
                    "score": w_score
                })

        return {
            "chunk_text": chunk_text,
            "avg_score": avg_score,               # 0..1 (UI đang dùng trực tiếp)
            "phonemes": phoneme_scores,           # [{phoneme, start, end, score, feedback}]
            "words": word_scores,                 # [{word, start, end, score, correct}]
        }

    except Exception as e:
        print("[ERROR] evaluate_pronunciation failed:", e)
        return {
            "chunk_text": "",
            "avg_score": 0.0,
            "phonemes": [],
            "words": [],
            "error": str(e)
        }