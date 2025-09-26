# ko_cleantext.py
# Normalize Hangul to "surface/pronunciation spelling" (clean text).
# Không dùng thư viện ngoài. Python 3.8+
#
# Điểm chính:
# - ㄴ-첨가 đặt TRƯỚC 연음화, nhưng:
#     (a) BỎ QUA khi coda là ㄷ/ㅌ và sau là 이 → dành cho liaison rồi 구개음화 (vd 같이→가치)
#     (b) BỎ QUA khi âm tiết sau là tiểu từ “이” ở cuối từ/nhóm (vd 옷이→오시)
#     (c) BỎ QUA khi coda có ㅎ (đơn/ghép) và sau là 이/야/여/요/유 (vd 많이→마니)
# - 구개음화:
#     (a) liaison tạo ㄷ/ㅌ + 이 → ㅈ/ㅊ (CHỈ tại ranh giới 연음)
#     (b) coda ㄷ/ㅌ + 히(ᄒ+ᅵ) → ᄎ (굳히다→구치다)
# - Mở rộng collapse_w_* (ㅘ/ㅙ/ㅚ/ㅝ/ㅞ/ㅟ)
# - Giới hạn 경음화 C tại ranh giới vừa liaison (để “읽어요→일거요”, vẫn “읽고→일꼬”).
# - Liaison với 겹받침 có ㅎ (ᆭ/ᆶ): rơi ㅎ, kéo đơn vị đầu (ㄴ/ㄹ) sang (vd 많이→마니, 밟아→발바).
# - API: normalize_clean(text, opt=None). opt có thể là CleanOptions hoặc dict.

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Union

# --- Hangul tables (Unicode) ---
S_BASE = 0xAC00
L_LIST = [chr(x) for x in range(0x1100, 0x1100+19)]
V_LIST = [chr(x) for x in range(0x1161, 0x1161+21)]
T_LIST = ["\0"] + [chr(x) for x in range(0x11A8, 0x11A8+27)]

CHO_TO_STR = {
    'ᄀ':'ㄱ','ᄁ':'ㄲ','ᄂ':'ㄴ','ᄃ':'ㄷ','ᄄ':'ㄸ','ᄅ':'ㄹ','ᄆ':'ㅁ',
    'ᄇ':'ㅂ','ᄈ':'ㅃ','ᄉ':'ㅅ','ᄊ':'ㅆ','ᄋ':'ㅇ','ᄌ':'ㅈ','ᄍ':'ㅉ',
    'ᄎ':'ㅊ','ᄏ':'ㅋ','ᄐ':'ㅌ','ᄑ':'ᄑ','ᄒ':'ㅎ'
}
JONG_TO_CHO = {
    'ᆨ':'ᄀ','ᆩ':'ᄁ','ᆪ':'ᄉ','ᆫ':'ᄂ','ᆬ':'ᄌ','ᆭ':'ᄒ','ᆮ':'ᄃ','ᆯ':'ᄅ',
    'ᆰ':'ᄀ','ᆱ':'ᄆ','ᆲ':'ᄇ','ᆳ':'ᄉ','ᆴ':'ᄐ','ᆵ':'ᄑ','ᆶ':'ᄒ','ᆷ':'ᄆ',
    'ᆸ':'ᄇ','ᆹ':'ᄉ','ᆺ':'ᄉ','ᆻ':'ᄊ','ᆼ':'ᄋ','ᆽ':'ᄌ','ᆾ':'ᄎ','ᆿ':'ᄏ',
    'ᇀ':'ᄐ','ᇁ':'ᄑ','ᇂ':'ᄒ'
}
TENSe_MAP = {'ᄀ':'ᄁ','ᄃ':'ᄄ','ᄇ':'ᄈ','ᄉ':'ᄊ','ᄌ':'ᄍ'}
ASP_MAP   = {'ᄀ':'ᄏ','ᄃ':'ᄐ','ᄇ':'ᄑ','ᄌ':'ᄎ'}

COMPLEX_CODA_SPLIT = {
    'ᆪ': ('ᆨ','ᆺ'), 'ᆬ': ('ᆫ','ᆽ'), 'ᆭ': ('ᆫ','ᇂ'), 'ᆰ': ('ᆯ','ᆨ'),
    'ᆱ': ('ᆯ','ᆷ'), 'ᆲ': ('ᆯ','ᆸ'), 'ᆳ': ('ᆯ','ᆺ'), 'ᆴ': ('ᆯ','ᇀ'),
    'ᆵ': ('ᆯ','ᇁ'), 'ᆶ': ('ᆯ','ᇂ'), 'ᆹ': ('ᆸ','ᆺ'),
}

K_LIKE_TAIL = {'ᆨ','ᆩ','ᆰ','ᆿ'}
T_LIKE_TAIL = {'ᆮ','ᆺ','ᆻ','ᆽ','ᆾ','ᇀ'}
P_LIKE_TAIL = {'ᆸ','ᆹ','ᇁ'}
H_TAIL      = {'ᇂ'}
N_TAIL      = {'ᆫ'}
M_TAIL      = {'ᆷ'}
L_TAIL      = {'ᆯ'}

V_IY_SET    = {'ᅵ','ᅣ','ᅧ','ᅭ','ᅲ'}

# w-collapse maps
W_SECOND_MAP = {'ᅪ':'ᅡ','ᅫ':'ᅢ','ᅬ':'ᅵ','ᅯ':'ᅥ','ᅰ':'ᅦ','ᅱ':'ᅵ'}
W_PERCEPT_MAP= {'ᅪ':'ᅡ','ᅫ':'ᅢ','ᅬ':'ᅦ','ᅯ':'ᅥ','ᅰ':'ᅦ','ᅱ':'ᅵ'}
V_WA        = 'ᅪ'  # legacy toggle

def is_hangul_syll(ch: str) -> bool:
    o = ord(ch)
    return 0xAC00 <= o <= 0xD7A3

@dataclass
class Syl:
    L: str; V: str; T: str; raw: str

def decompose(ch: str) -> Syl:
    sidx = ord(ch) - S_BASE
    L_idx = sidx // (21*28); V_idx = (sidx % (21*28)) // 28; T_idx = sidx % 28
    return Syl(L_LIST[L_idx], V_LIST[V_idx], (T_LIST[T_idx] if T_idx>0 else "\0"), ch)

def compose(s: Syl) -> str:
    L_idx = L_LIST.index(s.L); V_idx = V_LIST.index(s.V)
    T_idx = 0 if s.T == "\0" else T_LIST.index(s.T)
    return chr(S_BASE + (L_idx*21 + V_idx)*28 + T_idx)

def last_coda_unit(T: str) -> str:
    if T == "\0": return "\0"
    return COMPLEX_CODA_SPLIT[T][1] if T in COMPLEX_CODA_SPLIT else T

def first_coda_unit(T: str) -> str:
    return COMPLEX_CODA_SPLIT[T][0] if T in COMPLEX_CODA_SPLIT else "\0"

def set_coda(s: Syl, Tnew: str): s.T = Tnew
def set_onset(s: Syl, Lnew: str): s.L = Lnew
def has_vowel_onset(s: Syl) -> bool: return s.L == 'ᄋ'
def jong_to_cho(j: str) -> Optional[str]: return JONG_TO_CHO.get(j)
def tense(L: str) -> Optional[str]: return TENSe_MAP.get(L)
def aspirate(L: str) -> Optional[str]: return ASP_MAP.get(L)

# ---------- Options ----------
@dataclass
class CleanOptions:
    apply_across_space: bool = True
    collapse_wa_to_a: bool   = False            # legacy ㅘ→ㅏ
    collapse_w_mode: str     = "off"            # "off"|"second"|"perceptual"
    collapse_w_custom: Dict[str, str] = field(default_factory=dict)
    eui_reading: str         = "auto"           # "auto"|"keep"|"i"|"e"
    do_liaison: bool         = True
    do_n_insertion: bool     = True
    do_nasalization: bool    = True
    do_liquidization: bool   = True
    do_palatalization: bool  = True
    do_h_deletion: bool      = True
    do_h_aspiration: bool    = True
    do_tensification: bool   = True

def _collapse_w_vowel(v: str, opt: CleanOptions) -> Optional[str]:
    if v in opt.collapse_w_custom: return opt.collapse_w_custom[v]
    if opt.collapse_w_mode == "second": return W_SECOND_MAP.get(v)
    if opt.collapse_w_mode == "perceptual": return W_PERCEPT_MAP.get(v)
    if opt.collapse_w_mode == "off" and opt.collapse_wa_to_a and v == V_WA: return 'ᅡ'
    return None

def _is_particle_I(tokens: List[Tuple[Optional[Syl], str]], idx: int) -> bool:
    """
    Heuristic: coi '이' (ᄋ+ᅵ, no coda) là tiểu từ khi:
      - tokens[idx] là hangul '이' không 받침
      - và (idx là cuối cùng) hoặc (ký tự sau không phải Hangul)
    """
    if idx < 0 or idx >= len(tokens): return False
    sy, ty = tokens[idx]
    if ty != "H": return False
    if not (sy.L=='ᄋ' and sy.V=='ᅵ' and sy.T=="\0"): return False
    if idx == len(tokens)-1: return True
    return tokens[idx+1][1] != "H"

# ---------- Main ----------
def normalize_clean(
    text: str,
    opt: Optional[Union[CleanOptions, Dict[str, object]]] = None
) -> Tuple[str, List[str]]:
    """
    Normalize Hangul to a pronunciation-like spelling.

    opt:
      - None (mặc định): dùng CleanOptions() mặc định.
      - Dict[str, Any]: sẽ được dựng thành CleanOptions(**dict).
      - CleanOptions: dùng trực tiếp.
    """
    # Prepare options
    if opt is None:
        opt = CleanOptions()
    elif isinstance(opt, dict):
        opt = CleanOptions(**opt)  # type: ignore[call-arg]
    elif not isinstance(opt, CleanOptions):
        raise TypeError("opt must be None, dict, or CleanOptions")

    tokens: List[Tuple[Optional[Syl], str]] = []
    for ch in text:
        if is_hangul_syll(ch): tokens.append((decompose(ch), "H"))
        else: tokens.append((None, ch))

    rule_tags: List[str] = []
    # intra-syllable: collapse w*, 의
    for i,(sy, ty) in enumerate(tokens):
        if ty != "H": continue
        s = sy
        collapsed = _collapse_w_vowel(s.V, opt)
        if collapsed is not None and collapsed != s.V:
            old = s.V; s.V = collapsed; tokens[i]=(s,ty)
            rule_tags.append(f"collapse-w:{old}->{collapsed}@{i}[mode={opt.collapse_w_mode or 'legacy'}]")
        # 의 rule (very rough)
        if s.L=='ᄋ' and s.V=='ᅴ':
            if opt.eui_reading == "i" or (opt.eui_reading=="auto" and i>0 and tokens[i-1][1]=="H"):
                s.V='ᅵ'; tokens[i]=(s,ty); rule_tags.append(f"eui->i@{i}")
            elif opt.eui_reading == "e":
                s.V='ᅦ'; tokens[i]=(s,ty); rule_tags.append(f"eui->e@{i}")

    def boundary_allowed(_i:int)->bool: return True
    liaison_boundaries = set()

    for i in range(len(tokens)-1):
        if tokens[i][1]!="H" or tokens[i+1][1]!="H": continue
        if not boundary_allowed(i): continue
        A: Syl = tokens[i][0]; B: Syl = tokens[i+1][0]

        tail = last_coda_unit(A.T)
        first = first_coda_unit(A.T)

                # ===== 2) ㄴ-첨가 BEFORE liaison (tightened) =====
        if opt.do_n_insertion:
            # chỉ xét khi B có onset ᄋ (vowel onset) và là một trong {이, 야, 여, 요, 유} và KHÔNG có 받침
            is_IY_vowel = (B.L == 'ᄋ' and B.V in V_IY_SET and B.T == "\0")
            if A.T != "\0" and is_IY_vowel:
                tail = last_coda_unit(A.T)
                first = first_coda_unit(A.T)

                # 1) d/t + '이' -> dành cho liaison + 구개음화 (같이, 맏이 등)
                if (tail in {'ᆮ','ᇀ'}) and (B.V == 'ᅵ'):
                    pass  # skip ㄴ-첨가
                else:
                    # 2) '이' là tiểu từ ở cuối nhóm → skip
                    if _is_particle_I(tokens, i+1):
                        pass
                    else:
                        # 3) bất kỳ trường hợp coda có ㅎ (đơn/ghép) + I/Y → skip (để ㅎ-탈락/liaison riêng xử)
                        coda_has_h = (
                            tail == 'ᇂ' or
                            (A.T in COMPLEX_CODA_SPLIT and (COMPLEX_CODA_SPLIT[A.T][0] == 'ᇂ' or COMPLEX_CODA_SPLIT[A.T][1] == 'ᇂ'))
                        )
                        if coda_has_h:
                            pass
                        else:
                            # 4) nếu là 겹받침 nói chung → skip (nhường xử lý tách/liaison chuẩn)
                            if A.T in COMPLEX_CODA_SPLIT:
                                pass
                            else:
                                # 5) các coda “trơn” khác mới cân nhắc chèn
                                if tail in L_TAIL:
                                    set_onset(B, 'ᄅ')   # ㄹ + I/Y → ly
                                    rule_tags.append(f"n-insertion:ly@{i+1}")
                                else:
                                    set_onset(B, 'ᄂ')   # ㄴ-첨가
                                    rule_tags.append(f"n-insertion@{i+1}")
                                tokens[i+1] = (B, "H")

                                # refresh locals cho các bước sau
                                A = tokens[i][0]; B = tokens[i+1][0]
                                tail = last_coda_unit(A.T); first = first_coda_unit(A.T)

        # ===== 1) Liaison =====
        if opt.do_liaison and has_vowel_onset(B) and A.T!="\0":
            tail = last_coda_unit(A.T)
            first = first_coda_unit(A.T)

            # --- SPECIAL: 맛있다 / 멋있다 ... —
            # 패턴: (...)[coda ㅅ/ㅆ] + '있'(ᄋ+ᅵ+ᆻ) + '다'(ᄃ+ᅡ)
            # 결과 표면형: ... '싣'(ᄉ+ᅵ+ᆮ) + '따'(경음화)
            if (tail in {'ᆺ','ᆻ'}
                and B.L=='ᄋ' and B.V=='ᅵ' and B.T=='ᆻ'
                and i+1 < len(tokens)
                and tokens[i+1][1] == "H" and tokens[i+1][0].L == 'ᄃ'):
                # move ㅅ to onset of '있' → '시'
                set_onset(B, 'ᄉ')
                # keep '있'의 ㅆ as t-like coda → ㄷ
                set_coda(B, 'ᆮ')
                tokens[i+1] = (B, "H")

                # clear or reduce coda of previous syllable
                first_unit = first
                set_coda(A, first_unit if first_unit != "\0" else "\0")
                tokens[i] = (A, "H")

                rule_tags.append(f"special:masitda@{i+1}")
                liaison_boundaries.add(i)
                # (다음 경계 i+1에서 경음화 규칙이 ㄷ→ㄸ을 처리함)
                continue
        if opt.do_liaison and has_vowel_onset(B) and A.T!="\0":
            tail = last_coda_unit(A.T)
            first = first_coda_unit(A.T)

            # NEW: If complex coda whose second unit is ㅎ (e.g., ᆭ=ㄴㅎ, ᆶ=ㄹㅎ),
            # drop ㅎ and move the FIRST unit (ㄴ/ㄹ) to onset. This yields: 많이→마니, 밟아→발바.
            if tail == 'ᇂ':
                # skip moving ㅎ to onset (좋아요 → 조아요)
                pass
            else:
                is_complex_h_second = (A.T in COMPLEX_CODA_SPLIT and COMPLEX_CODA_SPLIT[A.T][1]=='ᇂ')
                if is_complex_h_second and first != "\0":
                    move_cho = jong_to_cho(first)
                    if move_cho:
                        # consume entire complex: drop ㅎ and move first; coda becomes empty
                        set_coda(A, "\0")
                        set_onset(B, move_cho)
                        tokens[i]=(A,"H"); tokens[i+1]=(B,"H")
                        rule_tags.append(f"liaison:complex-h(drop ㅎ, move {first})->{i+1}")
                        liaison_boundaries.add(i)
                else:
                    move_cho = jong_to_cho(tail)

                    # ❗Không liên kết 받침 ᆼ (không có onset /ŋ/ trong tiếng Hàn)
                    # và cũng không liên kết nếu move_cho == 'ᄋ' (onset câm).
                    if tail == 'ᆼ' or move_cho == 'ᄋ':
                        pass  # giữ nguyên: không chuyển 받침
                    else:
                        if move_cho:
                            first_unit = first
                            set_coda(A, first_unit if first_unit!="\0" else "\0")
                            set_onset(B, move_cho)
                            tokens[i]=(A,"H"); tokens[i+1]=(B,"H")
                            rule_tags.append(f"liaison:{i}->{i+1}")
                            liaison_boundaries.add(i)

        # Refresh
        if tokens[i][1]!="H" or tokens[i+1][1]!="H": continue
        A = tokens[i][0]; B = tokens[i+1][0]
        tail = last_coda_unit(A.T)
        first = first_coda_unit(A.T)

        # ===== 5a) 구개음화: onset ㄷ/ㅌ + 이 -> ㅈ/ㅊ (오직 '연음' 경계에서만) =====
        # 예: 같이(갇+이)→가치 ✔ / 어디(…디)→어지 ✘
        if (opt.do_palatalization
            and B.V=='ᅵ'
            and B.L in {'ᄃ','ᄐ'}
            and i in liaison_boundaries):   # ★ chỉ khi có liaison trước đó
            set_onset(B, 'ᄌ' if B.L=='ᄃ' else 'ᄎ')
            tokens[i+1]=(B,"H"); rule_tags.append(f"palatalize:(d/t)+i@{i+1}")

        # ===== 5b) 구개음화: coda ㄷ/ㅌ + 히(ᄒ+ᅵ) -> ᄎ =====
        if opt.do_palatalization and tail in {'ᆮ','ᇀ'} and B.L=='ᄒ' and B.V=='ᅵ':
            set_onset(B, 'ᄎ')
            first_unit = first
            set_coda(A, first_unit if first_unit!="\0" else "\0")
            tokens[i]=(A,"H"); tokens[i+1]=(B,"H")
            rule_tags.append(f"palatalize:(d/t)+hi->chi@{i+1}")

        # ===== 3) 비음화 =====
        if opt.do_nasalization:
            if B.L in {'ᄂ','ᄆ'}:
                if tail in K_LIKE_TAIL:
                    set_coda(A,'ᆼ'); tokens[i]=(A,"H"); rule_tags.append(f"nasal:k->ng@{i}")
                elif tail in T_LIKE_TAIL:
                    set_coda(A,'ᆫ'); tokens[i]=(A,"H"); rule_tags.append(f"nasal:t->n@{i}")
                elif tail in P_LIKE_TAIL:
                    set_coda(A,'ᆷ'); tokens[i]=(A,"H"); rule_tags.append(f"nasal:p->m@{i}")
            tail = last_coda_unit(A.T)
            if tail in M_TAIL.union({'ᆼ'}) and B.L=='ᄅ':
                set_onset(B,'ᄂ'); tokens[i+1]=(B,"H"); rule_tags.append(f"nasal:m/ng + l -> n@{i+1}")
            if tail in K_LIKE_TAIL.union(P_LIKE_TAIL) and B.L=='ᄅ':
                set_onset(B,'ᄂ'); tokens[i+1]=(B,"H"); rule_tags.append(f"nasal:k/p + l -> n@{i+1}")

        # ===== 4) 유음화 =====
        if opt.do_liquidization:
            tail = last_coda_unit(A.T)
            if tail=='ᆫ' and B.L=='ᄅ':
                set_coda(A,'ᆯ'); set_onset(B,'ᄅ')
                tokens[i]=(A,"H"); tokens[i+1]=(B,"H"); rule_tags.append(f"liquid:n+l->ll@{i}")
            elif tail=='ᆯ' and B.L=='ᄂ':
                set_onset(B,'ᄅ'); tokens[i+1]=(B,"H"); rule_tags.append(f"liquid:l+n->ll@{i+1}")

        # ===== 6) ㅎ-탈락 & 7) 격음화 =====
        if opt.do_h_deletion or opt.do_h_aspiration:
            tail = last_coda_unit(A.T)

            # 6a) ㅎ-탈락: coda = ㅎ và sau là nguyên âm
            if opt.do_h_deletion and has_vowel_onset(B) and tail == 'ᇂ':
                # 'first' là đơn vị đầu của coda gốc; với ᆭ/ᆶ sẽ là ᆫ/ᆯ
                if first != "\0":  # tức là coda gốc là 겹받침 có ㅎ như ᆭ/ᆶ
                    move_cho = jong_to_cho(first)
                    if move_cho:
                        set_coda(A, "\0")
                        set_onset(B, move_cho)
                        tokens[i]   = (A, "H")
                        tokens[i+1] = (B, "H")
                        rule_tags.append(f"liaison:complex-h(drop ㅎ, move {first})->{i+1}")
                        liaison_boundaries.add(i)
                else:
                    # coda chỉ là ㅎ đơn (ᇂ) ⇒ chỉ xóa ㅎ (좋아요→조아요)
                    first_unit = first  # '\0'
                    set_coda(A, first_unit if first_unit != "\0" else "\0")
                    tokens[i] = (A, "H")
                    rule_tags.append(f"h-deletion@{i}")

            # 6b) onset có ㅎ: h+onset -> aspirated onset (giữ quy tắc cũ)
            if opt.do_h_aspiration and tail=='ᇂ' and B.L in {'ᄀ','ᄃ','ᄇ','ᄌ'}:
                asp = aspirate(B.L)
                if asp:
                    set_onset(B, asp); tokens[i+1]=(B,"H")
                    first_unit = first
                    set_coda(A, first_unit if first_unit!="\0" else "\0")
                    tokens[i]=(A,"H"); rule_tags.append(f"h+onset->asp@{i+1}")

            tail = last_coda_unit(A.T)

            # 6c) NEW — coda + ㅎ(히) → aspirated onset (K/T/P) + liaison; cập nhật lại coda A
            # - K-like tails (…ᆨ류)  → ᄏ (막히다→마키다)
            # - T-like tails (…ᆮ/ᇀ류)→ ᄐ
            # - P-like tails (…ᆸ류)  → ᄑ (밟히다→발피다)
            if opt.do_h_aspiration and B.L=='ᄒ' and B.V=='ᅵ' and tail in K_LIKE_TAIL.union(T_LIKE_TAIL).union(P_LIKE_TAIL):
                if tail in K_LIKE_TAIL:
                    new_onset = 'ᄏ'
                elif tail in T_LIKE_TAIL:
                    new_onset = 'ᄐ'
                else:
                    new_onset = 'ᄑ'
                set_onset(B, new_onset)          # ㅋ/ㅌ/ㅍ
                tokens[i+1] = (B, "H")
                first_unit = first
                set_coda(A, first_unit if first_unit!="\0" else "\0")
                tokens[i] = (A, "H")
                rule_tags.append(f"coda+h(i)->asp+liaison@{i+1}")

        # ===== 8) 경음화 =====
        if opt.do_tensification and B.L in {'ᄀ','ᄃ','ᄇ','ᄉ','ᄌ'}:
            tL = tense(B.L)
            if tL:
                if last_coda_unit(A.T) in K_LIKE_TAIL.union(T_LIKE_TAIL).union(P_LIKE_TAIL):
                    set_onset(B, tL); tokens[i+1]=(B,"H"); rule_tags.append(f"tensify:A@{i+1}")
                elif last_coda_unit(A.T) in N_TAIL.union(M_TAIL) and B.L in {'ᄀ','ᄃ','ᄌ'}:
                    set_onset(B, tL); tokens[i+1]=(B,"H"); rule_tags.append(f"tensify:B@{i+1}")
                elif (last_coda_unit(A.T) in L_TAIL and B.L in {'ᄀ','ᄃ','ᄉ','ᄌ'}
                      and i not in liaison_boundaries):
                    set_onset(B, tL); tokens[i+1]=(B,"H"); rule_tags.append(f"tensify:C@{i+1}")

    out = []
    for sy, ty in tokens:
        out.append(compose(sy) if ty=="H" else ty)
    return "".join(out), rule_tags

# ===== User-facing explanation (no API break) =====
import re

def _only_hangul(s: str) -> list:
    return [ch for ch in s if is_hangul_syll(ch)]

def explain_clean_for_user(text: str, opt=None):
    """
    Trả:
      text_clean: chuỗi đã clean
      events: List[{
        "type": "liaison|n-insertion|nasalization|liquidization|palatalization|h-deletion|aspiration|tensification|collapse-w|eui",
        "at": int,                   # chỉ số ranh giới/âm tiếtjong_to_cho
        "before": str,               # lát cắt trước (2 âm tiết nếu là rule giữa ranh giới)
        "after": str,                # lát cắt sau
        "note": str,                 # ghi chú ngắn (vd 'ㄷ/ㅌ+이→ㅈ/ㅊ', 'ᆶ: drop ㅎ')
        "tag": str                   # raw tag để debug
      }, ...]
    """
    text_clean, rule_tags = normalize_clean(text, opt)
    ref = _only_hangul(text)
    cln = _only_hangul(text_clean)

    def _pair(ref_list, i):
        l = ref_list[i]   if 0 <= i < len(ref_list)     else ""
        r = ref_list[i+1] if 0 <= i+1 < len(ref_list)   else ""
        return l, r

    events = []
    for tag in rule_tags:
        # 1) Liaison (kể cả complex-h)
        m = re.match(r"liaison:(?:complex-h.*->)?(\d+)", tag)
        if m:
            i = int(m.group(1)) - 1 if "->" in tag and "complex-h" in tag else int(re.search(r"liaison:(\d+)->", tag).group(1)) if "->" in tag else int(m.group(1))
        if tag.startswith("liaison:"):
            m2 = re.search(r"(\d+)->(\d+)", tag)
            if m2:
                i = int(m2.group(1))
                lb, rb = _pair(ref, i)
                la, ra = _pair(cln, i)
                note = "연음화"
                if "complex-h" in tag:
                    note += " (겹받침 ㅎ 소실: ㄴㅎ/ㄹㅎ)"
                events.append({
                    "type":"liaison", "at": i,
                    "before": f"{lb}+{rb}", "after": f"{la}+{ra}",
                    "note": note, "tag": tag
                })
            continue

        # 2) ㄴ-첨가
        m = re.search(r"n-insertion@(\d+)", tag)
        if m:
            j = int(m.group(1))
            lb, rb = _pair(ref, j-1)
            la, ra = _pair(cln, j-1)
            events.append({
                "type":"n-insertion", "at": j-1,
                "before": f"{lb}+{rb}", "after": f"{la}+{ra}",
                "note":"ㄴ-첨가", "tag": tag
            })
            continue

        # 3) 비음화 (nasalization)
        m = re.search(r"nasal:([a-z]+)->([a-z]+)@(\d+)", tag)
        if m:
            i = int(m.group(3))
            lb, rb = _pair(ref, i)
            la, ra = _pair(cln, i)
            events.append({
                "type":"nasalization", "at": i,
                "before": f"{lb}+{rb}", "after": f"{la}+{ra}",
                "note": f"비음화 {m.group(1)}→{m.group(2)}", "tag": tag
            })
            continue

        # 4) 유음화 (liquidization)
        m = re.search(r"liquid:(.+)@(\d+)", tag)
        if m:
            i = int(m.group(2))
            lb, rb = _pair(ref, i)
            la, ra = _pair(cln, i)
            events.append({
                "type":"liquidization", "at": i,
                "before": f"{lb}+{rb}", "after": f"{la}+{ra}",
                "note":"유음화 (n/l 동화)", "tag": tag
            })
            continue

        # 5) 구개음화 (palatalization)
        if tag.startswith("palatalize:(d/t)+i"):
            m = re.search(r"@(\d+)", tag)
            i = int(m.group(1)) - 1 if m else 0
            lb, rb = _pair(ref, i)
            la, ra = _pair(cln, i)
            events.append({
                "type":"palatalization", "at": i,
                "before": f"{lb}+{rb}", "after": f"{la}+{ra}",
                "note":"구개음화: ㄷ/ㅌ + 이 → ㅈ/ㅊ", "tag": tag
            })
            continue
        if "palatalize:(d/t)+hi->chi" in tag:
            m = re.search(r"@(\d+)", tag)
            i = int(m.group(1)) - 1 if m else 0
            lb, rb = _pair(ref, i)
            la, ra = _pair(cln, i)
            events.append({
                "type":"palatalization", "at": i,
                "before": f"{lb}+{rb}", "after": f"{la}+{ra}",
                "note":"구개음화: 받침 ㄷ/ㅌ + 히 → ㅊ", "tag": tag
            })
            continue

        # 6) ㅎ-탈락 (h deletion) & 격음화 (aspiration)
        if tag.startswith("h-deletion@"):
            i = int(re.search(r"@(\d+)", tag).group(1))
            lb, rb = _pair(ref, i)
            la, ra = _pair(cln, i)
            events.append({
                "type":"h-deletion", "at": i,
                "before": f"{lb}+{rb}", "after": f"{la}+{ra}",
                "note":"ㅎ-탈락", "tag": tag
            })
            continue
        if "h+onset->asp@" in tag or "coda+h->asp@" in tag or "coda+h(i)->asp+liaison@" in tag:
            i = int(re.search(r"@(\d+)", tag).group(1)) - 1
            lb, rb = _pair(ref, i)
            la, ra = _pair(cln, i)
            events.append({
                "type":"aspiration", "at": i,
                "before": f"{lb}+{rb}", "after": f"{la}+{ra}",
                "note":"ㅎ 영향에 의한 격음화", "tag": tag
            })
            continue

        # 7) 경음화 (tensification)
        if tag.startswith("tensify:"):
            i = int(re.search(r"@(\d+)", tag).group(1)) - 1
            lb, rb = _pair(ref, i)
            la, ra = _pair(cln, i)
            events.append({
                "type":"tensification", "at": i,
                "before": f"{lb}+{rb}", "after": f"{la}+{ra}",
                "note":"경음화", "tag": tag
            })
            continue

        # 8) Intra-syllable: collapse-w, eui
        if tag.startswith("collapse-w:"):
            i = int(re.search(r"@(\d+)", tag).group(1))
            before = ref[i] if i < len(ref) else ""
            after  = cln[i] if i < len(cln) else ""
            events.append({
                "type":"collapse-w", "at": i,
                "before": before, "after": after,
                "note": tag.split("@")[0].replace("collapse-w:", "w-모음 정규화: "),
                "tag": tag
            })
            continue
        if tag.startswith("eui->"):
            i = int(re.search(r"@(\d+)", tag).group(1))
            before = ref[i] if i < len(ref) else ""
            after  = cln[i] if i < len(cln) else ""
            note = "의 발음: " + ("이" if "->i" in tag else "에")
            events.append({
                "type":"eui", "at": i,
                "before": before, "after": after,
                "note": note, "tag": tag
            })
            continue

    return text_clean, events

# ---------------- Demo ----------------
if __name__ == "__main__":
    tests = [
        ("같이", None),                # 가치 (liaison ㅌ + 이 → ㅊ)
        ("맏이", None),                # 마지
        ("굳이", None),                # 구지
        ("굳히다", None),              # 구치다 (d/t + 히)
        ("읽고", None),                # 일꼬
        ("읽어요", None),              # 일거요 (no tensify at liaison boundary)
        ("꽃잎", None),                # 꼰닙 (ㄴ-첨가 + 비음화)
        ("담요", None),                # 담뇨
        ("색연필", None),              # 생년필
        ("한여름", None),              # 한녀름
        ("옷이", None),                # 오시 (skip ㄴ-첨가 vì '이' là tiểu từ cuối)
        ("많이", None),                # 마니 (겹받침 ㄴㅎ: drop ㅎ, liaise ㄴ)
        ("밟아", None),                # 발바 (겹받침 ㄹㅎ: drop ㅎ, liaise ㄹ)
        ("어디 가요?", None),         # 구개음화 X → vẫn "어디 가요?"
        ("막히다", None),             # 마키다 (ㄱ + 히 → ㅋ)
        ("밟히다", None),             # 발피다 (ᆲ + 히 → ㄹ giữ, ㅂ→ㅍ onset)
        ("싫어", None),
        ("진짜 맛있다", None),
        ("자붐없이", None),
        ("다음 영상에서 만나요!", None),
        ("자붐없이", None),
    ]
    for txt, opt in tests:
        out, tags = explain_clean_for_user(txt, opt)
        notes_unique = list(dict.fromkeys(
                n.strip() for n in (e.get("note") for e in tags)
                if isinstance(n, str) and n.strip()
            ))
        print(f"{txt} -> {out}  // {notes_unique}")
