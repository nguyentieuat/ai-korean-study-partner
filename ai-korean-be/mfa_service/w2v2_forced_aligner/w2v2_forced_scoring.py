# w2v2_forced_scoring.py — CTC-only pronunciation scoring (Jamo-aware, robust gating)
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

# ===== Tunables =====
MISSING_THRESHOLD = float(os.getenv("W2V_MISSING_THRESHOLD", "0.08"))
MIN_SCORE_FLOOR = float(os.getenv("W2V_MIN_SCORE_FLOOR", "0.0"))   # floor=0 để tránh mặc định 45
ADVICE_THRESHOLD = float(os.getenv("W2V_ADVICE_THRESHOLD", "0.7"))
CTC_LOSS_CHAR_THRESHOLD = float(os.getenv("W2V_CTC_LOSS_CHAR_THRESHOLD", "2.0"))
CTC_CER_THRESHOLD = float(os.getenv("W2V_CTC_CER_THRESHOLD", "0.65"))
BLANK_VAD_THRESH = float(os.getenv("W2V_BLANK_VAD_THRESH", "0.85"))
VAD_PAD_MS = float(os.getenv("W2V_VAD_PAD_MS", "150.0"))
LOGIT_TEMP = float(os.getenv("W2V_LOGIT_TEMP", "0.5"))
ADVICE_JAMO_THRESHOLD = 0.75

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
    Set lam_edit=0 nếu muốn = mean(jamo)*100.
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
    - ᄋ onset: trả về {"silent":true, "conf":1.0, "score":100} và KHÔNG tính vào trung bình âm tiết.
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
    """
    Gắn advice theo từng jamo (bỏ qua jamo silent).
    """
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
    """
    Bỏ ᄋ onset khỏi danh sách dùng để tính điểm câu / missing-ratio.
    """
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
    ctc_loss_char_threshold: float = CTC_LOSS_CHAR_THRESHOLD,
    ctc_cer_threshold: float = CTC_CER_THRESHOLD,
    ctc_mix_mode: str = "ctc_only",
    tau_char_power: float = 1.0,
    lam_edit: float = 0.3,
) -> Dict[str, Any]:
    """
    Robust scoring với CTC gate:
    - Gate fail -> trả sàn điểm + breakdown=0.
    - Gate pass -> Viterbi-local jamo confidences + (tuỳ chọn) CER penalty.
    """
    assert (wav_path is not None) or (waveform is not None), "Provide wav_path or waveform"
    if waveform is None:
        import torchaudio
        waveform, waveform_sr = torchaudio.load(wav_path)

    # 1) Chuỗi jamo chuẩn (ưu tiên ref_chunks)
    flat_jamo = _flatten_ref_chunks(ref_chunks)
    if not flat_jamo:
        flat_jamo = _to_nfd_jamo_list(reference_text)
    n_ref = len([j for j in flat_jamo if j and not j.isspace()])

    # 2) Gate
    g = ctc_gate_global_from_waveform(
        waveform, waveform_sr, reference_text,
        cer_threshold=ctc_cer_threshold,
        loss_char_threshold=ctc_loss_char_threshold
    )
    # Tương thích mọi biến thể return
    cer_val = float(g.get("cer", 1.0))
    mnlp = float(g.get("mean_neglogp", 10.0))
    gate_pass = bool(
        g.get("gate_pass",
              g.get("pass_lenient",
                    (cer_val < ctc_cer_threshold and mnlp < ctc_loss_char_threshold)))
    )

    if not gate_pass and use_ctc_gate:
        # Gate fail: trả sàn + breakdown = 0
        zeros = [0.0] * n_ref
        return {
            "score": int(round(100.0 * min_score_floor)),
            "details": {
                "reason": "gate_fail",
                **g,
                "missing_ratio": 1.0,
                "lens": {"ref_jamo": n_ref, "ctc_char_confs": n_ref},
                "params": {
                    "tau_char_power": tau_char_power,
                    "lam_edit": lam_edit,
                    "blank_vad_thresh": BLANK_VAD_THRESH,
                    "vad_pad_ms": VAD_PAD_MS,
                }
            },
            "message": "Âm thanh không khớp câu tham chiếu (nghi ngờ nói bừa hoặc đọc sai nội dung).",
            "details_collapsed": _syllable_scores_from_chunks(ref_chunks, zeros, tau=tau_char_power)
        }

    # 3) Gate pass: per-jamo confidences
    char_confs = ctc_char_confidences_from_waveform(
        waveform, waveform_sr, reference_text,
        blank_vad_thresh=BLANK_VAD_THRESH, vad_pad_ms=VAD_PAD_MS,
        temp=LOGIT_TEMP, use_confusables=True
        # (không truyền ref_jamo_seq để giữ tương thích hàm hiện có)
    )

    # 3.1) Bỏ ᄋ onset khi tính điểm câu
    char_confs_eff = _filtered_for_sentence(char_confs, ref_chunks)

    # 4) Missing-ratio (sau khi lọc ᄋ onset)
    miss_ratio = float(np.mean([c < missing_threshold for c in char_confs_eff])) if len(char_confs_eff) > 0 else 1.0

    # 5) Breakdown + advice
    syllables = _syllable_scores_from_chunks(ref_chunks, char_confs, tau=tau_char_power)
    syllables = _attach_advices_to_syllables(syllables, thr=ADVICE_JAMO_THRESHOLD)

    # 6) Sentence score (đặt lam_edit=0 nếu muốn = mean jamo)
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
            },
            "score_formula": "power_mean(jamo_conf, p=tau) * (1 - lam_edit*CER) * 100"
        },
        "details_collapsed": syllables
    }
