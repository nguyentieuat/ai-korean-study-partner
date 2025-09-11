# score.py
# Scoring & advice for Korean G2P alignment (Python 3.8+)

from typing import List, Tuple, Dict, Optional, Any
import math

# ---- Import advice generator & vowel set ----
try:
    from utils.utils_advice import advices_for_phoneme, VOWELS as _VOWELS
except Exception:
    # Fallbacks (để không vỡ import khi tách môi trường)
    def advices_for_phoneme(curr, prev, next_, position, low_score=True, env_hint=None):
        return []
    _VOWELS = {"a","e","i","o","u","ɐ","ʌ","ɛ","ɯ","ø","y","ɨ","ə","ɤ","ʊ","œ","ɜ","æ","ʉ","ɒ","j","w"}

# ---- Optional TextGrid import (only used if you call textgrid_* helpers) ----
try:
    from textgrid import TextGrid
except Exception:
    TextGrid = None  # type: ignore

# ======================= Helpers =======================

def _is_vowel_ipa(ph: str) -> bool:
    return ph in _VOWELS

def _positions_in_chunk(phones: List[str]) -> List[Optional[str]]:
    """
    Heuristic cho vị trí âm trong một mảnh (label):
      - onset: trước nguyên âm đầu tiên
      - nucleus: nguyên âm (và glide j/w đi cùng nguyên âm)
      - coda: sau nguyên âm cuối
    """
    n = len(phones)
    if n == 0:
        return []
    first_v = None
    last_v = None
    for i, p in enumerate(phones):
        if _is_vowel_ipa(p):
            first_v = i
            break
    for i in range(n-1, -1, -1):
        if _is_vowel_ipa(phones[i]):
            last_v = i
            break
    pos: List[Optional[str]] = []
    for i, p in enumerate(phones):
        if first_v is None:
            pos.append("onset" if i == 0 else "coda")
        elif i < first_v:
            pos.append("onset")
        elif i > (last_v if last_v is not None else first_v):
            pos.append("coda")
        else:
            pos.append("nucleus" if (_is_vowel_ipa(p) or p in {"j","w"}) else "onset")
    return pos

def _flatten_ref_chunks_with_pos(
    ref_chunks: List[Tuple[str, List[str]]],
    drop_phones: Optional[set] = None
) -> Tuple[List[str], List[Tuple[str,int,int]], List[Optional[str]]]:
    """
    Chuyển [(label, [phones])...] -> ref_flat + spans + pos_map.
    - drop_phones: bỏ các phoneme tham chiếu (vd {"Ø"}).
    """
    drop_phones = drop_phones or set()
    ref_flat: List[str] = []
    spans: List[Tuple[str,int,int]] = []
    pos_map: List[Optional[str]] = []
    cur = 0
    for label, phones in ref_chunks:
        positions = _positions_in_chunk(phones)
        keep_idx = [i for i,p in enumerate(phones) if p not in drop_phones]
        start = cur
        for i in keep_idx:
            ref_flat.append(phones[i])
            pos_map.append(positions[i] if i < len(positions) else None)
        cur = len(ref_flat)
        spans.append((label, start, cur))
    return ref_flat, spans, pos_map

# ======================= Similarity =======================

# Tương đương mềm mặc định (có thể truyền eq_map riêng nếu muốn)
DEFAULT_EQUIV: Dict[Tuple[str, str], float] = {
    # coda unreleased ~ release
    ("k̚","k"):0.7, ("p̚","p"):0.7, ("t̚","t"):0.7,
    ("k","k̚"):0.7, ("p","p̚"):0.7, ("t","t̚"):0.7,
    # aspirated ~ lenis
    ("k","kʰ"):0.7, ("t","tʰ"):0.7, ("p","pʰ"):0.7, ("t͡ɕ","t͡ɕʰ"):0.7,
    ("kʰ","k"):0.7, ("tʰ","t"):0.7, ("pʰ","p"):0.7, ("t͡ɕʰ","t͡ɕ"):0.7,
    # tense vs lenis (thấp hơn)
    ("s","s͈"):0.6, ("t͡ɕ","t͡ɕ͈"):0.6, ("k","k͈"):0.6, ("t","t͈"):0.6, ("p","p͈"):0.6,
    ("s͈","s"):0.6, ("t͡ɕ͈","t͡ɕ"):0.6, ("k͈","k"):0.6, ("t͈","t"):0.6, ("p͈","p"):0.6,
    # liquid / flap
    ("l","ɾ"):0.7, ("ɾ","l"):0.7,
}

def _sim(a: str, b: str, eq_map: Optional[Dict[Tuple[str,str], float]], mismatch_penalty: float) -> float:
    if a == b:
        return 1.0
    if eq_map:
        if (a,b) in eq_map:
            return eq_map[(a,b)]
        if (b,a) in eq_map:
            return eq_map[(b,a)]
    return mismatch_penalty  # penalty âm cho mismatch

# ======================= Alignment (NW) =======================

def _nw_align(
    ref_seq: List[str],
    hyp_seq: List[str],
    *,
    gap_penalty: float = -0.3,
    mismatch_penalty: float = -0.4,
    eq_map: Optional[Dict[Tuple[str,str], float]] = None,
) -> Tuple[List[Tuple[Optional[int], Optional[int]]], float]:
    """
    Needleman–Wunsch (maximize).
    Trả (pairs, nw_path_score):
      - pairs: list cặp index (ri, hj):
          (ri,hj): match/mismatch
          (ri,None): deletion (ref)
          (None,hj): insertion (hyp)
      - nw_path_score: tổng điểm đường đi (dp[n][m])
    """
    n, m = len(ref_seq), len(hyp_seq)
    dp = [[0.0]*(m+1) for _ in range(n+1)]
    bt = [[0]*(m+1) for _ in range(n+1)]  # 0:diag, 1:up (del), 2:left (ins)

    for i in range(1, n+1):
        dp[i][0] = dp[i-1][0] + gap_penalty
        bt[i][0] = 1
    for j in range(1, m+1):
        dp[0][j] = dp[0][j-1] + gap_penalty
        bt[0][j] = 2

    for i in range(1, n+1):
        ai = ref_seq[i-1]
        for j in range(1, m+1):
            bj = hyp_seq[j-1]
            d = dp[i-1][j-1] + _sim(ai, bj, eq_map, mismatch_penalty)
            u = dp[i-1][j]   + gap_penalty
            l = dp[i][j-1]   + gap_penalty
            if d >= u and d >= l:
                dp[i][j] = d; bt[i][j] = 0
            elif u >= l:
                dp[i][j] = u; bt[i][j] = 1
            else:
                dp[i][j] = l; bt[i][j] = 2

    # Traceback
    i, j = n, m
    pairs: List[Tuple[Optional[int], Optional[int]]] = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and bt[i][j] == 0:
            pairs.append((i-1, j-1)); i -= 1; j -= 1
        elif i > 0 and (j == 0 or bt[i][j] == 1):
            pairs.append((i-1, None)); i -= 1
        else:
            pairs.append((None, j-1)); j -= 1
    pairs.reverse()
    return pairs, dp[n][m]

def _nw_bounds(
    n: int, m: int,
    *,
    match_score: float = 1.0,
    gap_penalty: float = -0.3,
    mismatch_penalty: float = -0.4
) -> Tuple[float, float]:
    """
    Trả (worst, best) để chuẩn hoá điểm NW:
      - best: min(n,m)*match + |n-m|*gap
      - worst: min(toàn mismatch + gap dư, toàn gap)
    """
    best = min(n, m)*match_score + abs(n - m)*gap_penalty
    worst_mis = min(n, m)*mismatch_penalty + abs(n - m)*gap_penalty
    worst_gap = n*gap_penalty + m*gap_penalty
    worst = min(worst_mis, worst_gap)
    return worst, best

# ======================= Diphthong collapse (post-process) =======================

# Cho phép gộp các cặp glide+vowel này (nếu cùng label và cùng nucleus)
_DIPHTHONG_PAIRS: Dict[Tuple[str,str], str] = {
    ("j","a"): "ja", ("j","ʌ"): "jʌ", ("j","o"): "jo", ("j","u"): "ju", ("j","e"): "je", ("j","ɛ"): "jɛ",
    ("w","a"): "wa", ("w","ʌ"): "wʌ", ("w","e"): "we", ("w","i"): "wi", ("w","ɛ"): "wɛ",
}

def _collapse_details_diphthongs(details: List[dict], score_mode: str = "avg") -> List[dict]:
    """
    Gộp (j|w)+V trong cùng 'label' và cả hai ở 'nucleus'.
    """
    out: List[dict] = []
    i = 0
    while i < len(details):
        d = details[i]
        if i + 1 < len(details):
            dn = details[i+1]
            if (d.get("label") == dn.get("label")
                and d.get("position") == "nucleus"
                and dn.get("position") == "nucleus"):
                pair = (d.get("phoneme"), dn.get("phoneme"))
                if pair in _DIPHTHONG_PAIRS:
                    merged_ph = _DIPHTHONG_PAIRS[pair]
                    if score_mode == "min":
                        merged_score = min(d.get("score", 0.0), dn.get("score", 0.0))
                    else:
                        merged_score = (d.get("score", 0.0) + dn.get("score", 0.0)) / 2.0
                    adv = list(dict.fromkeys((d.get("advice") or []) + (dn.get("advice") or [])))[:3]
                    mt = None
                    if d.get("matched") is not None or dn.get("matched") is not None:
                        mt = (d.get("matched") or "") + (dn.get("matched") or "")
                    dur = (d.get("duration") or 0.0) + (dn.get("duration") or 0.0)
                    out.append({
                        "label": d.get("label"),
                        "phoneme": merged_ph,
                        "matched": mt,
                        "score": merged_score,
                        "position": "nucleus",
                        "advice": adv,
                        "duration": dur if (d.get("duration") is not None or dn.get("duration") is not None) else None,
                    })
                    i += 2
                    continue
        out.append(d)
        i += 1
    return out

def _reaggregate_by_word_from_details(details: List[dict], *, duration_weight: str = "none") -> List[dict]:
    """
    Xây lại by_word từ details đã collapse (gom theo label liên tiếp).
    duration_weight: "none"|"linear"|"sqrt"|"log"
    """
    by_word: List[dict] = []
    if not details:
        return by_word
    cur_label = details[0]["label"]
    seg_ph, seg_scores, seg_adv, seg_dur = [], [], [], []
    for d in details:
        if d["label"] != cur_label:
            avg = sum(seg_scores)/len(seg_scores) if seg_scores else 0.0
            # weighted:
            wavg = avg
            if seg_dur and any(x is not None for x in seg_dur):
                weights = []
                for du in seg_dur:
                    if du is None:
                        weights.append(0.0)
                    else:
                        if duration_weight == "sqrt":
                            weights.append(math.sqrt(max(0.0, du)))
                        elif duration_weight == "log":
                            weights.append(math.log1p(max(0.0, du)))
                        elif duration_weight == "none":
                            weights.append(1.0)
                        else:  # linear
                            weights.append(max(0.0, du))
                tw = sum(weights)
                if tw > 0:
                    wavg = sum(s*w for s,w in zip(seg_scores, weights)) / tw
            by_word.append({
                "label": cur_label,
                "phonemes": seg_ph,
                "scores": seg_scores,
                "avg_score": avg,
                "avg_score_dur": wavg,
                "advice": list(dict.fromkeys(seg_adv))[:3],
            })
            cur_label = d["label"]
            seg_ph, seg_scores, seg_adv, seg_dur = [], [], [], []
        seg_ph.append(d["phoneme"])
        seg_scores.append(d["score"])
        seg_adv.extend(d.get("advice") or [])
        seg_dur.append(d.get("duration"))
    avg = sum(seg_scores)/len(seg_scores) if seg_scores else 0.0
    wavg = avg
    if seg_dur and any(x is not None for x in seg_dur):
        weights = []
        for du in seg_dur:
            if du is None:
                weights.append(0.0)
            else:
                if duration_weight == "sqrt":
                    weights.append(math.sqrt(max(0.0, du)))
                elif duration_weight == "log":
                    weights.append(math.log1p(max(0.0, du)))
                elif duration_weight == "none":
                    weights.append(1.0)
                else:
                    weights.append(max(0.0, du))
        tw = sum(weights)
        if tw > 0:
            wavg = sum(s*w for s,w in zip(seg_scores, weights)) / tw
    by_word.append({
        "label": cur_label,
        "phonemes": seg_ph,
        "scores": seg_scores,
        "avg_score": avg,
        "avg_score_dur": wavg,
        "advice": list(dict.fromkeys(seg_adv))[:3],
    })
    return by_word

# ======================= TextGrid utilities =======================

def _is_hangul_syllable_char(ch: str) -> bool:
    o = ord(ch)
    return 0xAC00 <= o <= 0xD7A3

def _tg_all_phones(
    tg_path: str,
    phone_tier: str = "phones",
    drop_phone_labels: Optional[set] = None,
    phone_aliases: Optional[Dict[str, str]] = None,
) -> Tuple[List[str], List[float]]:
    if TextGrid is None:
        raise RuntimeError("textgrid module is not available. Install 'praatio' or 'textgrid'.")
    tg = TextGrid.fromFile(tg_path)
    pt = None
    for t in tg.tiers:
        if t.name.lower() == phone_tier.lower():
            pt = t; break
    if pt is None:
        raise ValueError(f"Tier '{phone_tier}' not found in {tg_path}")
    drop = drop_phone_labels or {"sil", "sp", "pau", ""}
    aliases = phone_aliases or {}
    phones: List[str] = []
    durs:   List[float] = []
    for iv in pt:
        lab = (iv.mark or "").strip()
        if lab in drop:
            continue
        lab = aliases.get(lab, lab)
        phones.append(lab)
        durs.append(float(iv.maxTime) - float(iv.minTime))
    return phones, durs

def _segment_tg_phones_by_expected_nuclei(
    tg_phones: List[str],
    tg_durs: List[float],
    expected_nuclei: List[int],
) -> List[Tuple[List[str], List[float]]]:
    """
    Chia chuỗi phones của TextGrid thành len(expected_nuclei) segment.
    Mỗi segment mong có expected_nuclei[i] nguyên âm (thường =1 cho mỗi âm tiết).
    Quy tắc ranh giới:
      - nucleus = nguyên âm (theo _is_vowel_ipa), glide j/w dính với nguyên âm sau.
      - Khi đã đủ nucleus của segment hiện tại, vẫn “ăn” các phụ âm cuối (coda).
      - Bắt đầu segment tiếp theo khi nhìn-ahead thấy:
           + kế tiếp là j/w + vowel, hoặc
           + kế tiếp là vowel trực tiếp.
    Nếu TG còn dư phones sau segment cuối → dồn vào segment cuối.
    """
    segs: List[Tuple[List[str], List[float]]] = []
    n = len(tg_phones)
    i = 0
    for exp in expected_nuclei:
        phs: List[str] = []
        ds:  List[float] = []
        if exp <= 0:
            segs.append((phs, ds))
            continue
        nuc = 0
        while i < n:
            p = tg_phones[i]
            phs.append(p); ds.append(tg_durs[i])
            if _is_vowel_ipa(p):
                nuc += 1
            i += 1
            if nuc >= exp:
                if i >= n:
                    break
                nxt = tg_phones[i]
                if nxt in {"j", "w"}:
                    if (i + 1) < n and _is_vowel_ipa(tg_phones[i+1]):
                        break
                    else:
                        continue
                if _is_vowel_ipa(nxt):
                    break
        segs.append((phs, ds))
    if i < n and segs:
        segs[-1][0].extend(tg_phones[i:])
        segs[-1][1].extend(tg_durs[i:])
    return segs

def merge_textgrid_with_ref_chunks(
    tg_path: str,
    ref_chunks: List[Tuple[str, List[str]]],
    *,
    phone_tier: str = "phones",
    drop_phone_labels: Optional[set] = None,
    phone_aliases: Optional[Dict[str,str]] = None,
) -> Tuple[List[Tuple[str, List[str]]], List[float]]:
    """
    Dùng thứ tự & nhãn từ ref_chunks, nhưng phones/durations lấy từ TextGrid.
    - Tính số nucleus kỳ vọng cho mỗi mảnh từ ref_chunks (thường 1/mảnh).
    - Cắt chuỗi phone TG theo các nucleus đó.
    Trả:
      ref_chunks_tg: [(label, phones_from_TG)]
      durations_flat: durations theo thứ tự flatten của ref_chunks_tg
    """
    tg_phones, tg_durs = _tg_all_phones(
        tg_path, phone_tier=phone_tier,
        drop_phone_labels=drop_phone_labels, phone_aliases=phone_aliases
    )
    exp_nucs: List[int] = []
    for label, phs in ref_chunks:
        vn = sum(1 for p in (phs or []) if _is_vowel_ipa(p))
        if vn == 0 and any(_is_hangul_syllable_char(ch) for ch in label):
            vn = 1
        exp_nucs.append(vn)
    segs = _segment_tg_phones_by_expected_nuclei(tg_phones, tg_durs, exp_nucs)
    out_chunks: List[Tuple[str, List[str]]] = []
    durations_flat: List[float] = []
    for (label, _), (phs, ds) in zip(ref_chunks, segs):
        out_chunks.append((label, phs))
        durations_flat.extend(ds)
    return out_chunks, durations_flat

# ======================= TextGrid utilities (revised) =======================
def textgrid_read_phone_sequence(
    tg_path: str,
    *,
    phone_tier: str = "phones",
    drop_phone_labels: Optional[set] = None,
    phone_aliases: Optional[Dict[str,str]] = None,
) -> Tuple[List[str], List[float]]:
    """
    Đọc phone-tier -> (phones_seq, durations_seq).
    Không nhóm theo word. Dùng để align với ref_flat.
    """
    if TextGrid is None:
        raise RuntimeError("textgrid module is not available. Install 'praatio' or 'textgrid'.")

    tg = TextGrid.fromFile(tg_path)
    pt = None
    for t in tg.tiers:
        if t.name.lower() == phone_tier.lower():
            pt = t; break
    if pt is None:
        raise ValueError(f"Tier '{phone_tier}' not found in {tg_path}")

    drop = set(drop_phone_labels or set()) or set()
    # mở rộng noise mặc định
    drop.update({"sil","sp","pau","","spn","nsn","brth","noise","laugh","tsk"})

    phone_aliases = phone_aliases or {}

    phones: List[str] = []
    durs:   List[float] = []
    for iv in pt:
        lab_raw = (iv.mark or "").strip()
        if lab_raw in drop:
            continue
        lab = phone_aliases.get(lab_raw, lab_raw)
        dur = float(iv.maxTime) - float(iv.minTime)
        # bỏ các đoạn 0s
        if dur <= 0:
            continue
        phones.append(lab)
        durs.append(dur)
    return phones, durs


def durations_for_ref_via_alignment(
    ref_flat: List[str],
    tg_phones: List[str],
    tg_durations: List[float],
    *,
    gap_penalty: float = -0.3,
    mismatch_penalty: float = -0.4,
    eq_map: Optional[Dict[Tuple[str,str], float]] = None,
) -> List[Optional[float]]:
    """
    Align ref_flat với (tg_phones, tg_durations) → suy ra duration cho từng ref phoneme.
    - Nếu (ri,hj) match/mismatch: cộng dồn duration tg[hj] vào ref[ri].
    - Nếu (ri,None): duration None (không có phone TG tương ứng).
    - Nếu (None,hj): bỏ (chèn TG, không gán ref).
    """
    pairs, _ = _nw_align(
        ref_flat, tg_phones,
        gap_penalty=gap_penalty,
        mismatch_penalty=mismatch_penalty,
        eq_map=eq_map or DEFAULT_EQUIV
    )
    out: List[Optional[float]] = [None] * len(ref_flat)
    for ri, hj in pairs:
        if ri is not None and hj is not None:
            dur = tg_durations[hj]
            out[ri] = (out[ri] or 0.0) + dur
    return out

# ======================= Public API =======================

def score_chunks_vs_hyp(
    ref_chunks: List[Tuple[str, List[str]]],
    hyp_seq: List[str],
    *,
    drop_phones_ref: Optional[set] = None,
    drop_phones_hyp: Optional[set] = None,
    eq_map: Optional[Dict[Tuple[str,str], float]] = None,
    gap_penalty: float = -0.3,
    mismatch_penalty: float = -0.4,
    advice_threshold: float = 0.85,
    collapse_diphthongs: bool = False,
    diphthong_score_mode: str = "avg",
    # NEW: durations (seconds) cho từng ref-phoneme (đã align cùng ref_flat sau khi drop_phones_ref)
    ref_phone_durations: Optional[List[float]] = None,
    duration_weight: str = "none",   # "none"|"linear"|"sqrt"|"log"
) -> Dict[str, Any]:
    """
    ref_chunks: [(label, [phones])...] (từ return_by_word=True) hoặc build từ TextGrid.
    hyp_seq:    [phones] (ASR/MFA)
    Trả:
      {
        avg_score: float,                     # trung bình unweighted
        avg_score_dur: float | None,          # trung bình weighted theo duration (nếu có)
        details: [ {label, phoneme, matched, score, position, advice, duration?}, ... ],
        by_word: [ {label, phonemes, scores, avg_score, avg_score_dur?, advice}, ... ],
        ref_flat: [..], hyp_kept: [..], pairs: [(ri,hj)|...],
        nw_path_score: float, nw_norm_score: float, nw_params: {...},
        # nếu collapse_diphthongs:
        details_collapsed: [...], by_word_collapsed: [...]
      }
    """
    drop_phones_ref = drop_phones_ref or {"Ø"}
    drop_phones_hyp = drop_phones_hyp or {"Ø"}
    eq_map = eq_map or DEFAULT_EQUIV

    # 1) Flatten ref + spans + positions
    ref_flat, spans, pos_map = _flatten_ref_chunks_with_pos(ref_chunks, drop_phones=drop_phones_ref)
    hyp_kept = [p for p in hyp_seq if p not in drop_phones_hyp]

    # 1b) durations sanity (optional)
    dur_map: List[Optional[float]] = [None]*len(ref_flat)
    if ref_phone_durations:
        if len(ref_phone_durations) == len(ref_flat):
            dur_map = list(ref_phone_durations)
        else:
            dur_map = [None]*len(ref_flat)

    # 2) Align (lấy cả NW path score)
    pairs, nw_path_score = _nw_align(
        ref_flat, hyp_kept,
        gap_penalty=gap_penalty,
        mismatch_penalty=mismatch_penalty,
        eq_map=eq_map
    )
    worst, best = _nw_bounds(
        len(ref_flat), len(hyp_kept),
        match_score=1.0,
        gap_penalty=gap_penalty,
        mismatch_penalty=mismatch_penalty
    )
    if best - worst != 0:
        nw_norm_score = max(0.0, min(1.0, (nw_path_score - worst) / (best - worst)))
    else:
        nw_norm_score = 1.0

    # 3) Scores per ref-phoneme + matched
    per_ref_scores = [0.0]*len(ref_flat)
    matched_to: List[Optional[str]] = [None]*len(ref_flat)
    for (ri, hj) in pairs:
        if ri is not None and hj is not None:
            per_ref_scores[ri] = _sim(ref_flat[ri], hyp_kept[hj], eq_map, mismatch_penalty)
            matched_to[ri] = hyp_kept[hj]
        elif ri is not None:
            per_ref_scores[ri] = 0.0
            matched_to[ri] = None

    # 4) Advice per-phoneme
    details: List[dict] = []
    label_map: List[str] = []
    for (label, s, e) in spans:
        for _ in range(s, e):
            label_map.append(label)

    for i, (ph, sc) in enumerate(zip(ref_flat, per_ref_scores)):
        prev_ph = ref_flat[i-1] if i > 0 else None
        next_ph = ref_flat[i+1] if i+1 < len(ref_flat) else None
        pos = pos_map[i] if i < len(pos_map) else None
        mt = matched_to[i]
        env_hint = None
        if mt is not None and ph != mt:
            if ("͈" in ph) != ("͈" in mt):
                env_hint = "tense_diff"
            elif (ph.endswith("ʰ")) != (mt.endswith("ʰ")):
                env_hint = "asp_diff"
            elif (ph in {"k̚","t̚","p̚"}) != (mt in {"k̚","t̚","p̚"}):
                env_hint = "release_diff"
            elif (ph in {"j","w"}) != (mt in {"j","w"}):
                env_hint = "glide_diff"

        tips = advices_for_phoneme(
            curr=ph,
            prev=prev_ph,
            next_=next_ph,
            position=pos,
            low_score=(sc < advice_threshold),
            env_hint=env_hint
        )
        details.append({
            "label": label_map[i] if i < len(label_map) else "",
            "phoneme": ph,
            "matched": mt,
            "score": sc,
            "position": pos,
            "advice": tips,
            "duration": dur_map[i] if i < len(dur_map) else None,
        })

    # 5) Aggregate by label (by_word)
    by_word: List[dict] = []
    for (label, s, e) in spans:
        seg_scores = per_ref_scores[s:e]
        seg_ph = ref_flat[s:e]
        seg_details = details[s:e]
        avg = sum(seg_scores)/len(seg_scores) if seg_scores else 0.0

        # Weighted theo duration nếu có
        wavg = None
        seg_dur = [d.get("duration") for d in seg_details]
        if seg_dur and any(x is not None for x in seg_dur):
            weights = []
            for du in seg_dur:
                if du is None:
                    weights.append(0.0)
                else:
                    if duration_weight == "sqrt":
                        weights.append(math.sqrt(max(0.0, du)))
                    elif duration_weight == "log":
                        weights.append(math.log1p(max(0.0, du)))
                    elif duration_weight == "none":
                        weights.append(1.0)
                    else:
                        weights.append(max(0.0, du))
            tw = sum(weights)
            if tw > 0:
                wavg = sum(s*w for s,w in zip(seg_scores, weights)) / tw

        # gom advice (tối đa 3) từ các phoneme điểm thấp
        adv_agg: List[str] = []
        seen = set()
        for d in seg_details:
            if d["score"] < advice_threshold:
                for tip in d.get("advice") or []:
                    if tip not in seen:
                        seen.add(tip)
                        adv_agg.append(tip)
                    if len(adv_agg) >= 3:
                        break
            if len(adv_agg) >= 3:
                break
        by_word.append({
            "label": label,
            "phonemes": seg_ph,
            "scores": seg_scores,
            "avg_score": avg,
            "avg_score_dur": wavg,
            "advice": adv_agg
        })

    overall = sum(per_ref_scores)/len(per_ref_scores) if per_ref_scores else 0.0
    overall_dur = None
    if any(d is not None for d in dur_map):
        weights = []
        for du in dur_map:
            if du is None:
                weights.append(0.0)
            else:
                if duration_weight == "sqrt":
                    weights.append(math.sqrt(max(0.0, du)))
                elif duration_weight == "log":
                    weights.append(math.log1p(max(0.0, du)))
                elif duration_weight == "none":
                    weights.append(1.0)
                else:
                    weights.append(max(0.0, du))
        tw = sum(weights)
        if tw > 0:
            overall_dur = sum(s*w for s,w in zip(per_ref_scores, weights)) / tw

    result: Dict[str, Any] = {
        "avg_score": overall,
        "avg_score_dur": overall_dur,
        "details": details,
        "by_word": by_word,
        "ref_flat": ref_flat,
        "hyp_kept": hyp_kept,
        "pairs": pairs,
        "nw_path_score": nw_path_score,
        "nw_norm_score": nw_norm_score,
        "nw_params": {
            "gap_penalty": gap_penalty,
            "mismatch_penalty": mismatch_penalty
        },
    }

    # 6) Optional: collapse diphthongs (j/w + V) chỉ cho cách hiển thị
    if collapse_diphthongs:
        details_c = _collapse_details_diphthongs(details, score_mode=diphthong_score_mode)
        by_word_c = _reaggregate_by_word_from_details(details_c, duration_weight=duration_weight)
        result["details_collapsed"] = details_c
        result["by_word_collapsed"] = by_word_c

    return result

# ======================= Convenience: score directly with TextGrid =======================
def score_with_textgrid(
    textgrid_path: str,
    hyp_seq: List[str],
    *,
    ref_chunks: Optional[List[Tuple[str, List[str]]]] = None,  # bắt buộc nếu muốn mốc 젖|어|요
    word_tier: str = "words",      # giữ tham số cho tương thích, nhưng không dùng để cắt phones
    phone_tier: str = "phones",
    drop_phone_labels: Optional[set] = None,
    phone_aliases: Optional[Dict[str,str]] = None,
    drop_phones_ref: Optional[set] = None,
    drop_phones_hyp: Optional[set] = None,
    eq_map: Optional[Dict[Tuple[str,str], float]] = None,
    gap_penalty: float = -0.3,
    mismatch_penalty: float = -0.4,
    advice_threshold: float = 0.85,
    collapse_diphthongs: bool = False,
    diphthong_score_mode: str = "avg",
    duration_weight: str = "linear"   # khuyến nghị linear khi dùng duration
) -> Dict[str, Any]:
    """
    Đọc phone-tier → (phones, durations), ALIGN với ref_flat (từ ref_chunks),
    rồi chấm điểm so với hyp_seq. Như vậy ranh giới mảnh sẽ theo ref_chunks,
    không phụ thuộc word-tier của TextGrid.
    """
    if not ref_chunks:
        raise ValueError("score_with_textgrid: 'ref_chunks' is required to define the chunk boundaries (e.g., 젖|어|요).")

    # 1) Lấy chuỗi phones TG phẳng
    tg_phones, tg_durs = textgrid_read_phone_sequence(
        textgrid_path,
        phone_tier=phone_tier,
        drop_phone_labels=drop_phone_labels,
        phone_aliases=phone_aliases
    )

    # 2) Flatten ref để align durations
    ref_flat, spans, _pos_map = _flatten_ref_chunks_with_pos(ref_chunks, drop_phones=drop_phones_ref or {"Ø"})

    # 3) Durations cho ref theo alignment TG↔ref
    ref_phone_durations = durations_for_ref_via_alignment(
        ref_flat, tg_phones, tg_durs,
        gap_penalty=gap_penalty,
        mismatch_penalty=mismatch_penalty,
        eq_map=eq_map or DEFAULT_EQUIV
    )

    # 4) Chấm điểm với hyp_seq (ASR/MFA), dùng durations vừa map
    return score_chunks_vs_hyp(
        ref_chunks, hyp_seq,
        drop_phones_ref=drop_phones_ref,
        drop_phones_hyp=drop_phones_hyp,
        eq_map=eq_map,
        gap_penalty=gap_penalty,
        mismatch_penalty=mismatch_penalty,
        advice_threshold=advice_threshold,
        collapse_diphthongs=collapse_diphthongs,
        diphthong_score_mode=diphthong_score_mode,
        ref_phone_durations=ref_phone_durations,
        duration_weight=duration_weight
    )

# ======================= Quick self-test =======================
if __name__ == "__main__":
    # Ví dụ tối giản (không cần TextGrid)
    ref_chunks = [
        ("놔", ["n","w","a"]),
        ("써", ["s͈","ʌ"]),
        ("요",  ["j","o"]),
    ]
    hyp_seq = ["n","w","a","s͈","ʌ","j","o"]

    out = score_chunks_vs_hyp(
        ref_chunks, hyp_seq,
        gap_penalty = -0.2,
        mismatch_penalty = -0.5,
        advice_threshold = 0.85,
        collapse_diphthongs = True,
        diphthong_score_mode = "avg",
        ref_phone_durations = [0.08,0.04,0.07, 0.05,0.09, 0.04,0.06],
        duration_weight = "linear"
    )
    print("overall_avg:", round(out["avg_score"], 4))
    print("overall_avg_dur:", round(out["avg_score_dur"] or 0.0, 4))
    print("nw_path_score:", round(out["nw_path_score"], 4))
    print("nw_norm_score:", round(out["nw_norm_score"], 4))
    print("by_word:", [(x["label"], round(x["avg_score"],3), round((x.get('avg_score_dur') or 0.0),3)) for x in out["by_word"]])
    if "by_word_collapsed" in out:
        print("by_word_collapsed:", [(x["label"], x["phonemes"]) for x in out["by_word_collapsed"]])
