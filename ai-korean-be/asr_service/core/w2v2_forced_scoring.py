"""
CTC-only pronunciation scoring (Jamo-aware, robust gating).
- Tự nhận diện vocab mode ở ctc_gate.py (NFD Jamo / Compatibility Jamo / Âm tiết).
- Pad 0.2s đầu/đuôi cho clip ngắn (≤~0.6s) để tránh VAD cắt mất biên.
- Produces per-jamo/per-syllable confidences; advice cho jamo thấp.
- Nếu có TextGrid, dùng duration priors per-jamo để bias alignment.
"""

# w2v2_forced_scoring.py — CTC-only pronunciation scoring (Jamo-aware, robust gating)
# Python 3.9+

import os
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model

from core.ctc_gate import (
    ctc_gate_global_from_waveform,
    ctc_char_confidences_from_waveform,
)
from utils.utils_advice import advices_for_jamo

# ===== Globals for encoder feature extraction =====
ENC_MODEL_NAME = os.getenv("W2V_ENCODER_MODEL", os.getenv("W2V_MODEL", "Kkonjeong/wav2vec2-base-korean"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR = 16000

_processor2: Optional[Wav2Vec2Processor] = None
_encoder: Optional[Wav2Vec2Model] = None

def init_w2v2() -> None:
    global _processor2, _encoder
    if _processor2 is None or _encoder is None:
        _processor2 = Wav2Vec2Processor.from_pretrained(ENC_MODEL_NAME)
        _encoder = Wav2Vec2Model.from_pretrained(ENC_MODEL_NAME).to(DEVICE)
        _encoder.eval()

def preload_forced_encoder(device: Optional[str] = None, model_name: Optional[str] = None):
    """Preload encoder (Wav2Vec2Model) để warm-up."""
    global ENC_MODEL_NAME, DEVICE
    if model_name:
        ENC_MODEL_NAME = model_name
    if device:
        DEVICE = device
    init_w2v2()
    # warm-up 1 forward ngắn
    import torch
    sr = TARGET_SR
    T = int(sr * 0.6)
    wav = torch.zeros(1, T)
    _ = extract_features_from_waveform(wav, sr)

@torch.inference_mode()
def extract_features_from_waveform(waveform: torch.Tensor, sr: int) -> Tuple[torch.Tensor, int, float]:
    init_w2v2()
    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        import torchaudio
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        sr = TARGET_SR
    inputs = _processor2(waveform.squeeze(0), sampling_rate=sr, return_tensors="pt", padding=False)
    input_values = inputs.input_values.to(DEVICE)
    out = _encoder(input_values)
    h = out.last_hidden_state[0].detach().cpu()
    total_sec = float(waveform.shape[-1]) / sr
    return h, sr, total_sec

# ===== Tunables (đồng bộ với ctc_gate.py) =====
MISSING_THRESHOLD = float(os.getenv("W2V_MISSING_THRESHOLD", "0.08"))
MIN_SCORE_FLOOR = float(os.getenv("W2V_MIN_SCORE_FLOOR", "0.0"))
ADVICE_THRESHOLD = float(os.getenv("W2V_ADVICE_THRESHOLD", "0.7"))
W2V_CTC_REF_NLL_THR = float(os.getenv("W2V_CTC_REF_NLL_THR", "2.2"))
CTC_LOSS_CHAR_THRESHOLD = float(os.getenv("W2V_CTC_LOSS_CHAR_THRESHOLD", "2.0"))
CTC_CER_THRESHOLD = float(os.getenv("W2V_CTC_CER_THRESHOLD", "0.65"))
BLANK_VAD_THRESH = float(os.getenv("W2V_BLANK_VAD_THRESH", "0.8"))
VAD_PAD_MS = float(os.getenv("W2V_VAD_PAD_MS", "520.0"))
LOGIT_TEMP = float(os.getenv("W2V_LOGIT_TEMP", "1.8"))
ADVICE_JAMO_THRESHOLD = float(os.getenv("W2V_ADVICE_JAMO_THRESHOLD", "0.75"))

SILENT_ONSET = "\u110B"  # 'ᄋ'

# ===== Helpers =====
def _to_nfd_jamo_list(text: str) -> List[str]:
    return list(text)

def _flatten_ref_chunks(ref_chunks: Optional[List[Tuple[str, List[str]]]]) -> List[str]:
    flat: List[str] = []
    if not ref_chunks:
        return flat
    for _, jamos in ref_chunks:
        if jamos:
            flat.extend([j for j in jamos if j])
    return flat

def _power_mean(xs: List[float], p: float = 1.0) -> float:
    xs = np.clip(np.asarray(xs, dtype=np.float32), 0.0, 1.0)
    if xs.size == 0:
        return 0.0
    if p == 1.0:
        return float(xs.mean())
    return float((xs ** p).mean() ** (1.0 / p))

def _score_sentence(char_confs: List[float], cer: float, tau: float = 1.0, lam_edit: float = 0.3, n_ref_chars: int = 0) -> float:
    """
    Overall = power_mean(char_confs, p=tau) * (1 - lam_edit * CER) * 100, floored bởi MIN_SCORE_FLOOR.
    Với chuỗi cực ngắn (n_ref_chars<=2) giảm nhẹ lam_edit để ít phạt.
    """
    base = _power_mean(char_confs, p=tau)
    lam = min(lam_edit, 0.1) if n_ref_chars <= 2 else lam_edit
    score = base * (1.0 - min(1.0, lam * cer))
    return max(MIN_SCORE_FLOOR, float(100.0 * score))

def _is_silent_onset(j: str, j_idx: int) -> bool:
    return j_idx == 0 and j == SILENT_ONSET

def _syllable_scores_from_chunks(
    ref_chunks: Optional[List[Tuple[str, List[str]]]],
    char_confs: List[float],
    tau: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    - ᄋ onset: {"silent":true, "conf":1.0, "score":100} và KHÔNG tính vào trung bình âm tiết.
    """
    out: List[Dict[str, Any]] = []
    idx = 0
    n = len(char_confs)

    for syl_text, jamos in (ref_chunks or []):
        jamo_items: List[Dict[str, Any]] = []
        vals: List[float] = []

        for j_idx, j in enumerate(jamos or []):
            conf = float(char_confs[idx]) if idx < n else 0.0
            is_silent = _is_silent_onset(j, j_idx)

            shown_conf = 1.0 if is_silent else conf
            jamo_items.append({
                "jamo": j,
                "conf": shown_conf,
                "score": int(round(100.0 * np.clip(shown_conf, 0.0, 1.0))),
                "silent": bool(is_silent),
            })
            if not is_silent:
                vals.append(conf)
            idx += 1

        syl_conf = _power_mean(vals, p=tau) if len(vals) > 0 else 0.0
        out.append({
            "text": syl_text,
            "text_nfd": "".join(jamos or []),
            "conf": float(syl_conf),
            "score": int(round(100.0 * syl_conf)),
            "jamo": jamo_items,
        })

    # spillover (nếu CTC trả nhiều hơn)
    if idx < n:
        spill = char_confs[idx:]
        syl_conf = _power_mean(spill, p=tau)
        out.append({
            "text": "",
            "text_nfd": "",
            "conf": float(syl_conf),
            "score": int(round(100.0 * syl_conf)),
            "jamo": [
                {"jamo": "", "conf": float(c), "score": int(round(100.0 * np.clip(c, 0.0, 1.0)))}
                for c in spill
            ],
            "note": "spillover_extra_ctc_chars"
        })
    return out

def _attach_advices_to_syllables(
    syllables: List[Dict[str, Any]],
    thr: float = ADVICE_JAMO_THRESHOLD
) -> List[Dict[str, Any]]:
    """Gắn advice theo từng jamo (bỏ qua jamo silent)."""
    for syl in syllables:
        jlist = syl.get("jamo", [])
        advice_syl = []
        for idx, j in enumerate(jlist):
            if j.get("silent"):
                continue
            conf = float(j.get("conf", 0.0))
            if conf >= thr:
                continue
            pos = "onset" if idx == 0 else ("nucleus" if idx == 1 else "coda")
            curr = str(j.get("jamo", "")) or None
            prev = str(jlist[idx-1].get("jamo","")) if idx-1 >= 0 else None
            next_ = str(jlist[idx+1].get("jamo","")) if idx+1 < len(jlist) else None
            tips = advices_for_jamo(curr, prev, next_, pos, low_score=True, env_hint=None)
            if tips:
                j["advice"] = tips
                for t in tips:
                    if t not in advice_syl:
                        advice_syl.append(t)
        if advice_syl:
            syl["advice"] = advice_syl
    return syllables

def _filtered_for_sentence(confs: List[float], ref_chunks: Optional[List[Tuple[str, List[str]]]]) -> List[float]:
    """Bỏ ᄋ onset khỏi danh sách dùng để tính điểm câu/missing-ratio."""
    if not ref_chunks:
        return confs
    keep, idx = [], 0
    for _, jamos in ref_chunks:
        for j_idx, j in enumerate(jamos or []):
            if not _is_silent_onset(j, j_idx):
                if idx < len(confs):
                    keep.append(confs[idx])
            idx += 1
    return keep if keep else confs

# ===== TextGrid helpers (optional) =====
def _load_textgrid_intervals(path: str) -> List[Tuple[str, float, float]]:
    """
    Try to read a Praat TextGrid file. Returns list of (label, start, end) intervals.
    Prefers a tier named 'syllable'/'phones'; otherwise the first non-empty interval tier.
    """
    try:
        import textgrid  # praat-textgrids
        tg = textgrid.TextGrid.fromFile(path)
        tier = None
        for name in ["syllable", "Syllable", "syll", "phones", "phone", "segments"]:
            try:
                tier = tg.getFirst(name)
                if tier: break
            except Exception:
                continue
        if tier is None:
            for t in tg.tiers:
                if getattr(t, "intervals", None) and len(t.intervals) > 0:
                    tier = t; break
        if tier is None:
            return []
        out = []
        for iv in tier.intervals:
            mark = (iv.mark or "").strip()
            out.append((mark, float(iv.minTime), float(iv.maxTime)))
        return [(m,s,e) for (m,s,e) in out if e>s]
    except Exception:
        pass
    try:
        import tgt  # textgrid (tgt)
        tg = tgt.io.read_textgrid(path)
        tier = None
        for name in ["syllable", "Syllable", "syll", "phones", "phone", "segments"]:
            tier = tg.get_tier_by_name(name) if tg.has_tier(name) else None
            if tier: break
        if tier is None:
            tiers = tg.get_tier_names()
            if tiers:
                tier = tg.get_tier_by_name(tiers[0])
        if tier is None:
            return []
        out = []
        for iv in tier.intervals:
            mark = (iv.text or "").strip()
            out.append((mark, float(iv.start_time), float(iv.end_time)))
        return [(m,s,e) for (m,s,e) in out if e>s]
    except Exception:
        pass
    # Naive fallback
    try:
        lines = open(path, "r", encoding="utf-8").read().splitlines()
    except Exception:
        return []
    intervals = []
    cur = {}
    for ln in lines:
        ls = ln.strip()
        if ls.startswith("intervals ["):
            cur = {}
        elif ls.startswith("xmin ="):
            cur["xmin"] = float(ls.split("=")[1].strip())
        elif ls.startswith("xmax ="):
            cur["xmax"] = float(ls.split("=")[1].strip())
        elif ls.startswith("text ="):
            txt = ls.split("=",1)[1].strip().strip('"')
            cur["text"] = txt
            if "xmin" in cur and "xmax" in cur:
                intervals.append((cur.get("text",""), cur["xmin"], cur["xmax"]))
                cur = {}
    return [(m,s,e) for (m,s,e) in intervals if e>s]

def _priors_from_textgrid_for_chunks(ref_chunks: Optional[List[Tuple[str, List[str]]]],
                                     tg_path: Optional[str]) -> Optional[List[Tuple[float,float]]]:
    """
    Map TextGrid intervals → per-jamo (flattened). Nếu phone-level = len(flat_jamo) khớp → dùng trực tiếp.
    Nếu chỉ có syllable-level = len(syllables) → chia đều cho jamo trong âm tiết.
    """
    if not tg_path or not os.path.exists(tg_path):
        return None
    intervals = _load_textgrid_intervals(tg_path)
    if not intervals:
        return None

    # flatten jamo
    flat = []
    by_syl = []
    for syl_text, jamos in (ref_chunks or []):
        arr = [j for j in (jamos or []) if j]
        by_syl.append(arr)
        flat.extend(arr)

    phones_like = [(m,s,e) for (m,s,e) in intervals if (e>s)]
    if len(phones_like) == len(flat):
        return [(s,e) for (_,s,e) in phones_like]

    if len(intervals) == len(by_syl) and len(by_syl) > 0:
        priors: List[Tuple[float,float]] = []
        for (_label, s, e), jamos in zip(intervals, by_syl):
            if not jamos:
                continue
            dur = (e - s) / len(jamos)
            for i in range(len(jamos)):
                js = s + i * dur
                je = js + dur
                priors.append((js, je))
        return priors

    return None

# ===== Public =====
def score_pronunciation_forced(
    wav_path: Optional[str] = None,
    waveform: Optional[torch.Tensor] = None,
    waveform_sr: Optional[int] = None,
    vectors_user: Any = None,
    reference_text: str = "",              # expected NFD Jamo string
    ref_chunks: Optional[List[Tuple[str, List[str]]]] = None,  # [(syllable, [jamo...]), ...]
    ref_textgrid_path: Optional[str] = None,
    char_confidences: Optional[List[float]] = None,  # ignored
    missing_threshold: float = MISSING_THRESHOLD,
    min_score_floor: float = MIN_SCORE_FLOOR,
    advice_threshold: float = ADVICE_THRESHOLD,
    use_ctc_gate: bool = True,
    ref_nll_threshold: float = W2V_CTC_REF_NLL_THR,
    ctc_loss_char_threshold: float = CTC_LOSS_CHAR_THRESHOLD,
    ctc_cer_threshold: float = CTC_CER_THRESHOLD,
    ctc_mix_mode: str = "ctc_only",
    tau_char_power: float = 1.0,
    lam_edit: float = 0.3,
) -> Dict[str, Any]:
    """
    Robust scoring với CTC gate:
    - Gate fail -> score chính thức = 0 nhưng vẫn tính breakdown + score_soft (cap thấp).
    - Gate pass -> Viterbi-local jamo/syllable confidences + (tuỳ chọn) CER penalty.
    - Nếu có TextGrid, dùng duration priors per jamo khi align.
    """
    assert (wav_path is not None) or (waveform is not None), "Provide wav_path or waveform"

    import torchaudio
    if waveform is None:
        waveform, waveform_sr = torchaudio.load(wav_path)

    # Pad 0.2s đầu/đuôi cho clip ngắn (≤~0.6s)
    sr_eff = waveform_sr or TARGET_SR
    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if waveform.shape[-1] < int(0.6 * sr_eff):
        pad = torch.zeros(1, int(0.2 * sr_eff))
        waveform = torch.cat([pad, waveform, pad], dim=-1)

    # 1) Chuỗi jamo chuẩn (ưu tiên ref_chunks)
    flat_jamo = _flatten_ref_chunks(ref_chunks)
    if not flat_jamo:
        flat_jamo = _to_nfd_jamo_list(reference_text)
    n_ref = len([j for j in flat_jamo if j and not j.isspace()])

    # 2) Gate
    g = ctc_gate_global_from_waveform(
        waveform, waveform_sr or TARGET_SR, reference_text,
        cer_threshold=ctc_cer_threshold,
        loss_char_threshold=ctc_loss_char_threshold,
        ref_nll_threshold=ref_nll_threshold
    )
    cer_val = float(g.get("cer", 1.0))
    mnlp = float(g.get("mean_neglogp", 10.0))
    gate_pass = bool(
        g.get("gate_pass",
              g.get("pass_lenient",
                    (cer_val < ctc_cer_threshold and mnlp < ctc_loss_char_threshold)))
    )

    if not gate_pass and use_ctc_gate:
        # Gate fail: score chính thức = 0, vẫn tính breakdown + soft score (cap)
        soft_cap = int(float(os.getenv("W2V_SOFT_CAP", "35")))

        priors_sec = _priors_from_textgrid_for_chunks(ref_chunks, ref_textgrid_path)
        char_confs_raw = ctc_char_confidences_from_waveform(
            waveform, waveform_sr or TARGET_SR, reference_text,
            blank_vad_thresh=BLANK_VAD_THRESH, vad_pad_ms=VAD_PAD_MS,
            temp=LOGIT_TEMP, use_confusables=True,
            priors_sec=priors_sec
        )

        char_confs_eff = _filtered_for_sentence(char_confs_raw, ref_chunks)
        miss_ratio = float(np.mean([c < missing_threshold for c in char_confs_eff])) if len(char_confs_eff) > 0 else 1.0

        soft = _score_sentence(
            char_confs_eff, cer_val,
            tau=tau_char_power, lam_edit=lam_edit,
            n_ref_chars=len(char_confs_eff)
        )
        soft = min(int(round(soft)), soft_cap)

        syllables = _syllable_scores_from_chunks(ref_chunks, char_confs_raw, tau=tau_char_power)
        syllables = _attach_advices_to_syllables(syllables, thr=ADVICE_JAMO_THRESHOLD)

        len_mismatch = (len(_flatten_ref_chunks(ref_chunks)) != len(char_confs_raw))

        return {
            "score": int(round(100.0 * min_score_floor)),  # điểm chính thức = 0
            "score_soft": soft,                            # tham khảo
            "gate_pass": False,
            "details": {
                "reason": "gate_fail",
                **g,
                "missing_ratio": miss_ratio,
                "len_mismatch": bool(len_mismatch),
                "lens": {
                    "ref_jamo": len(_flatten_ref_chunks(ref_chunks)),
                    "ctc_char_confs": len(char_confs_raw),
                },
                "params": {
                    "tau_char_power": tau_char_power,
                    "lam_edit": lam_edit,
                    "blank_vad_thresh": BLANK_VAD_THRESH,
                    "vad_pad_ms": VAD_PAD_MS,
                    "soft_cap": soft_cap,
                    "logit_temp": LOGIT_TEMP,
                },
                "note": "Gate fail: score (chính thức) = 0; score_soft là tham khảo để góp ý phát âm."
            },
            "message": "Bạn chưa đọc đúng câu tham chiếu (điểm chính thức = 0). Dưới đây là góp ý phát âm tham khảo.",
            "details_collapsed": syllables
        }

    # 3) Gate pass → per-unit confidences
    priors_sec = _priors_from_textgrid_for_chunks(ref_chunks, ref_textgrid_path)
    char_confs = ctc_char_confidences_from_waveform(
        waveform, waveform_sr or TARGET_SR, reference_text,
        blank_vad_thresh=BLANK_VAD_THRESH, vad_pad_ms=VAD_PAD_MS,
        temp=LOGIT_TEMP, use_confusables=True,
        priors_sec=priors_sec
    )

    # 3.1) Bỏ ᄋ onset khi tính điểm câu
    char_confs_eff = _filtered_for_sentence(char_confs, ref_chunks)

    # 4) Missing-ratio
    miss_ratio = float(np.mean([c < missing_threshold for c in char_confs_eff])) if len(char_confs_eff) > 0 else 1.0

    # 5) Breakdown + advice
    syllables = _syllable_scores_from_chunks(ref_chunks, char_confs, tau=tau_char_power)
    syllables = _attach_advices_to_syllables(syllables, thr=ADVICE_JAMO_THRESHOLD)

    # 6) Sentence score
    score = _score_sentence(char_confs_eff, cer_val, tau=tau_char_power, lam_edit=lam_edit, n_ref_chars=len(char_confs_eff))

    # 7) Length check
    len_mismatch = (len(flat_jamo) != len(char_confs))

    return {
        "score": int(round(score)),
        "details": {
            **g,
            "missing_ratio": miss_ratio,
            "len_mismatch": bool(len_mismatch),
            "lens": {
                "ref_jamo": len(flat_jamo),
                "ctc_char_confs": len(char_confs),
            },
            "params": {
                "tau_char_power": tau_char_power,
                "lam_edit": lam_edit,
                "blank_vad_thresh": BLANK_VAD_THRESH,
                "vad_pad_ms": VAD_PAD_MS,
                "logit_temp": LOGIT_TEMP,
            },
            "score_formula": "power_mean(jamo_conf, p=tau) * (1 - lam_edit*CER) * 100"
        },
        "details_collapsed": syllables
    }
