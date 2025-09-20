# w2v2_forced_scoring_optimized.py — optimized CTC-only pronunciation scoring
# Adds init_w2v2() and extract_features_from_waveform() for drop-in compatibility
# Uses ref_chunks to produce per-syllable and per-jamo scores
# Gate-fail now "scores by wrongness": jamo below threshold -> 0, above -> keep
# No dependency on 'editdistance'
# Python 3.9+

import os
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model

from w2v2_forced_aligner.ctc_gate import (
    ctc_gate_global_from_waveform,
    ctc_char_confidences_from_waveform,
)

# ===== Globals for encoder feature extraction =====
ENC_MODEL_NAME = os.getenv("W2V_ENCODER_MODEL", os.getenv("W2V_MODEL", "kresnik/wav2vec2-large-xlsr-korean"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR = 16000

_processor2: Optional[Wav2Vec2Processor] = None
_encoder: Optional[Wav2Vec2Model] = None

def init_w2v2() -> None:
    """Idempotent init of processor+encoder for feature extraction (no CTC head)."""
    global _processor2, _encoder
    if _processor2 is None or _encoder is None:
        _processor2 = Wav2Vec2Processor.from_pretrained(ENC_MODEL_NAME)
        _encoder = Wav2Vec2Model.from_pretrained(ENC_MODEL_NAME).to(DEVICE)
        _encoder.eval()

@torch.inference_mode()
def extract_features_from_waveform(waveform: torch.Tensor, sr: int) -> Tuple[torch.Tensor, int, float]:
    """Return (features[T,D], sr, total_sec). Features are last_hidden_state on CPU."""
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
    h = out.last_hidden_state[0].detach().cpu()  # [T,D]
    total_sec = float(waveform.shape[-1]) / sr
    return h, sr, total_sec

# ===== Tunables (can be overridden by function args) =====
MISSING_THRESHOLD = float(os.getenv("W2V_MISSING_THRESHOLD", "0.08"))
MIN_SCORE_FLOOR = float(os.getenv("W2V_MIN_SCORE_FLOOR", "0.45"))
ADVICE_THRESHOLD = float(os.getenv("W2V_ADVICE_THRESHOLD", "0.7"))
CTC_LOSS_CHAR_THRESHOLD = float(os.getenv("W2V_CTC_LOSS_CHAR_THRESHOLD", "2.0"))
CTC_CER_THRESHOLD = float(os.getenv("W2V_CTC_CER_THRESHOLD", "0.65"))
BLANK_VAD_THRESH = float(os.getenv("W2V_BLANK_VAD_THRESH", "0.7"))
VAD_PAD_MS = float(os.getenv("W2V_VAD_PAD_MS", "120.0"))
LOGIT_TEMP = float(os.getenv("W2V_LOGIT_TEMP", "1.0"))

# NEW: jamo pass threshold when gate-fail
FAIL_JAMO_THRESH = float(os.getenv("W2V_FAIL_JAMO_THRESH", "0.5"))

# ===== Helpers =====
def _to_nfd_jamo_list(text: str) -> List[str]:
    """Assume input is already NFD Jamo; just explode to a char list."""
    return list(text)

def _flatten_ref_chunks(ref_chunks: Optional[List[Tuple[str, List[str]]]]) -> List[str]:
    """
    Flatten [(syllable_text, [j1, j2, ...]), ...] -> [j1, j2, ...,]
    Used for length checks vs. CTC char confidences.
    """
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
    Overall sentence score = power-mean of char confidences * (1 - lam_edit * CER), floored.
    For very short refs (<=2 jamo), limit penalty weight slightly to avoid harsh down-scaling.
    """
    base = _power_mean(char_confs, p=tau)
    lam = min(lam_edit, 0.1) if n_ref_chars <= 2 else lam_edit
    score = base * (1.0 - min(1.0, lam * cer))
    return max(MIN_SCORE_FLOOR, float(100.0 * score))

def _syllable_scores_from_chunks(
    ref_chunks: Optional[List[Tuple[str, List[str]]]],
    char_confs: List[float],
    tau: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Walk through ref_chunks and consume confidences per jamo, producing per-syllable and per-jamo scores.
    Also appends a spillover item if CTC confidences are longer than total jamo from ref_chunks.
    """
    out: List[Dict[str, Any]] = []
    idx = 0
    n = len(char_confs)

    for syl_text, jamos in (ref_chunks or []):
        jamo_items: List[Dict[str, Any]] = []
        vals: List[float] = []

        for j in (jamos or []):
            conf = float(char_confs[idx]) if idx < n else 0.0
            jamo_items.append({
                "jamo": j,
                "conf": conf,
                "score": int(round(100.0 * np.clip(conf, 0.0, 1.0))),
            })
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

    # Handle extra confidences if any (mismatch between G2P chunks and CTC timesteps/tokens)
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

# ===== Public =====
def score_pronunciation_forced(
    wav_path: Optional[str] = None,
    waveform: Optional[torch.Tensor] = None,
    waveform_sr: Optional[int] = None,
    vectors_user: Any = None,
    reference_text: str = "",              # expected NFD Jamo
    ref_chunks: Optional[List[Tuple[str, List[str]]]] = None,  # [(syllable, [jamo...]), ...]
    ref_textgrid_path: Optional[str] = None,  # unused
    char_confidences: Optional[List[float]] = None,  # ignored (CTC-only)
    missing_threshold: float = MISSING_THRESHOLD,
    min_score_floor: float = MIN_SCORE_FLOOR,
    advice_threshold: float = ADVICE_THRESHOLD,
    use_ctc_gate: bool = True,
    ctc_loss_char_threshold: float = CTC_LOSS_CHAR_THRESHOLD,
    ctc_cer_threshold: float = CTC_CER_THRESHOLD,
    ctc_mix_mode: str = "ctc_only",
    tau_char_power: float = 1.0,
    lam_edit: float = 0.3,
    # NEW:
    fail_jamo_threshold: float = FAIL_JAMO_THRESH,
) -> Dict[str, Any]:
    """
    Compute pronunciation score using CTC-only confidences + mild CER penalty.
    Now aligns scoring breakdown to ref_chunks for per-jamo output.
    When gate-fail, we still compute jamo scores but zero-out jamo whose conf < fail_jamo_threshold.
    """
    assert (wav_path is not None) or (waveform is not None), "Provide wav_path or waveform"
    if waveform is None:
        import torchaudio
        waveform, waveform_sr = torchaudio.load(wav_path)

    # 1) Ref lengths
    nfd_chars = _to_nfd_jamo_list(reference_text)
    n_ref = len(nfd_chars)

    # 2) Optional global CTC gate signals
    cer_gate = 0.0
    mean_neglogp = 0.0
    gate_flag = False
    if use_ctc_gate:
        g = ctc_gate_global_from_waveform(
            waveform, waveform_sr, reference_text,
            cer_threshold=ctc_cer_threshold,
            loss_char_threshold=ctc_loss_char_threshold
        )
        cer_gate = float(g.get("cer", 0.0))
        mean_neglogp = float(g.get("mean_neglogp", 0.0))
        # mark as fail but DO NOT early-return — we still compute jamo-level scores
        if n_ref > 2 and cer_gate > 0.95 and mean_neglogp > 5.0:
            gate_flag = True

    # 3) Per-char CTC confidences over reference jamo stream
    char_confs = ctc_char_confidences_from_waveform(
        waveform, waveform_sr, reference_text,
        blank_vad_thresh=BLANK_VAD_THRESH,
        vad_pad_ms=VAD_PAD_MS,
        temp=LOGIT_TEMP,
        use_confusables=True
    )

    # 4) If gate-fail: zero-out wrong jamo (confidence below threshold)
    if gate_flag:
        char_confs_eff = [c if c >= fail_jamo_threshold else 0.0 for c in char_confs]
    else:
        char_confs_eff = char_confs

    # 5) Missing-ratio (per jamo) for debugging/UX — compute on effective confidences
    miss_ratio = float(np.mean([c < missing_threshold for c in char_confs_eff])) if n_ref > 0 else 1.0

    # 6) Per-syllable & per-jamo scores aligned to ref_chunks
    syllables = _syllable_scores_from_chunks(ref_chunks, char_confs_eff, tau=tau_char_power)

    # 7) Sentence-level score with mild CER penalty
    score_raw = _score_sentence(char_confs_eff, cer_gate, tau=tau_char_power, lam_edit=lam_edit, n_ref_chars=n_ref)
    # If gate failed, clamp to floor to avoid super-low UI shock
    score_out = int(round(max(100.0 * MIN_SCORE_FLOOR, score_raw))) if gate_flag else int(round(score_raw))

    # 8) Length checks (handy for debugging tokenizer/G2P mismatches)
    flat_from_chunks = _flatten_ref_chunks(ref_chunks)
    len_mismatch = (len(flat_from_chunks) != len(char_confs))

    details: Dict[str, Any] = {
        "cer": cer_gate,
        "mean_neglogp": mean_neglogp,
        "missing_ratio": miss_ratio,
        "len_mismatch": bool(len_mismatch),
        "lens": {
            "ref_jamo_from_chunks": len(flat_from_chunks),
            "ctc_char_confs": len(char_confs),
        },
        "params": {
            "tau_char_power": tau_char_power,
            "lam_edit": lam_edit,
            "blank_vad_thresh": BLANK_VAD_THRESH,
            "vad_pad_ms": VAD_PAD_MS,
            "fail_jamo_threshold": fail_jamo_threshold,
        }
    }
    if gate_flag:
        details["reason"] = "gate_fail_scored"  # we didn't early-return; scored-by-wrongness

    return {
        "score": score_out,
        "details": details,
        "details_collapsed": syllables
    }
