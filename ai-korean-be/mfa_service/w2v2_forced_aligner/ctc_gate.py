
# ctc_gate.py — CTC-only gate & confidences for Korean (NFD Jamo)
# - Blank-posterior VAD (no energy)
# - Viterbi-local CTC for per-jamo confidences (Jamo vocab only)
# - Optional time priors from TextGrid (bias Viterbi with per-jamo windows)
# - Lenient gate for 1–2 syllables
# - Optional logit temperature & confusable sets
# Python 3.9+

import os
import logging
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

_log = logging.getLogger(__name__)
if not _log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

# ============ Config ============
# NOTE: MODEL_NAME must point to a Jamo-CTC checkpoint (vocab = NFD Jamo + space, [PAD] as blank)
MODEL_NAME = os.getenv("W2V_MODEL", "Kkonjeong/wav2vec2-base-korean")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR = 16000
VAD_PAD_MS = float(os.getenv("W2V_VAD_PAD_MS", "120.0"))
BLANK_VAD_THRESH = float(os.getenv("W2V_BLANK_VAD_THRESH", "0.7"))
LOGIT_TEMP = float(os.getenv("W2V_LOGIT_TEMP", "1.0"))

# Bias when a frame is outside the token's prior window (log-space)
PRIOR_OUT_PENALTY = float(os.getenv("W2V_PRIOR_OUT_PENALTY", "3.0"))  # subtract this from logprob (i.e., add -penalty)
PRIOR_EDGE_PAD_SEC = float(os.getenv("W2V_PRIOR_EDGE_PAD_SEC", "0.02"))

# Globals (lazy)
_processor: Optional[Wav2Vec2Processor] = None
_model: Optional[Wav2Vec2ForCTC] = None
_blank_id: int = 0

# ============ Utils ============
def _ensure_sr(waveform: torch.Tensor, sr: int) -> torch.Tensor:
    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    return waveform

def _init():
    global _processor, _model, _blank_id
    if _processor is None or _model is None:
        _log.info(f"Loading W2V2 model (Jamo CTC): {MODEL_NAME}")
        _processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        _model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(DEVICE)
        _model.eval()
        _blank_id = int(getattr(_model.config, "pad_token_id", 0) or 0)

def _log_softmax(logits: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
    if temp != 1.0:
        logits = logits / max(1e-6, temp)
    return torch.log_softmax(logits, dim=-1)

def _blank_posterior_vad(logprobs: torch.Tensor, blank_id: int, thresh: float = 0.7, pad_ms: float = 120.0) -> slice:
    """Return time slice [start:end] to crop voiced region using blank posterior threshold."""
    with torch.no_grad():
        blank_post = logprobs.exp()[:, blank_id]  # [T]
        non_blank = (blank_post < thresh).nonzero(as_tuple=False).squeeze(-1)
        if non_blank.numel() == 0:
            return slice(0, logprobs.shape[0])  # keep all
        start = non_blank[0].item()
        end = non_blank[-1].item() + 1
        # Pad by ms in terms of frames (approximate: assume ~20ms/frame)
        pad_frames = int(round(pad_ms / 20.0))
        start = max(0, start - pad_frames)
        end = min(logprobs.shape[0], end + pad_frames)
        return slice(start, end)

# ============ Jamo helpers ============
def _to_nfd_jamo(text: str) -> List[str]:
    """Assume input already normalized to NFD Jamo upstream; keep as-is."""
    return list(text)

# Confusable sets (light) — only for leading consonants; optional bonus mass
CONFUSABLES: Dict[str, List[str]] = {
    "ᄂ": ["ᄅ"], "ᄅ": ["ᄂ"],
    "ᄇ": ["ᄑ"], "ᄑ": ["ᄇ"],
    "ᄃ": ["ᄐ", "ᄌ", "ᄎ"], "ᄐ": ["ᄃ", "ᄌ", "ᄎ"],
    "ᄌ": ["ᄃ", "ᄐ", "ᄎ"], "ᄎ": ["ᄃ", "ᄐ", "ᄌ"],
}

def _expand_ref_with_blanks(ref_ids: List[int], blank_id: int) -> List[int]:
    out = []
    for tid in ref_ids:
        out.append(blank_id)
        out.append(tid)
    out.append(blank_id)
    return out

# ============ Viterbi-local CTC ============
def _forced_ctc_viterbi_path(logprobs: torch.Tensor, ext: List[int], bias_ext: Optional[torch.Tensor]=None) -> torch.Tensor:
    """Viterbi on extended target ext; returns best state index m per frame.
    If bias_ext is provided (shape [T,M]), it is added to logprobs[:, ext] before DP (time priors)."""
    T, M = logprobs.shape[0], len(ext)
    dp = torch.full((T, M), -1e9, device=logprobs.device)
    bp = torch.full((T, M), -1, device=logprobs.device, dtype=torch.long)

    step0 = logprobs[0, ext]
    if bias_ext is not None:
        step0 = step0 + bias_ext[0]
    dp[0, 0] = step0[0]
    if M > 1:
        dp[0, 1] = step0[1]
        bp[0, 1] = 0

    for t in range(1, T):
        stay = dp[t-1]
        prev = torch.cat([torch.tensor([-1e9], device=dp.device), dp[t-1, :-1]], dim=0)
        prev2 = torch.cat([torch.tensor([-1e9, -1e9], device=dp.device), dp[t-1, :-2]], dim=0)
        # disallow same-token skip
        mask = torch.ones(M, dtype=torch.bool, device=dp.device)
        if M >= 3:
            mask[2:] = torch.tensor([ext[m] != ext[m-2] for m in range(2, M)], device=dp.device)
        prev2 = torch.where(mask, prev2, torch.full_like(prev2, -1e9))
        cand = torch.stack([stay, prev, prev2], 0)  # [3,M]
        best, arg = cand.max(0)

        step = logprobs[t, ext]
        if bias_ext is not None:
            step = step + bias_ext[t]
        dp[t] = best + step
        bp[t] = arg  # 0: stay, 1: -1, 2: -2

    # backtrack
    m = M-1 if M > 1 else 0
    if M > 1 and dp[T-1, M-2] > dp[T-1, M-1]:
        m = M-2
    m_path = torch.empty(T, dtype=torch.long, device=dp.device)
    m_path[-1] = m
    for t in range(T-2, -1, -1):
        move = bp[t+1, m]
        if move == 1:
            m -= 1
        elif move == 2:
            m -= 2
        m_path[t] = m
    return m_path

def _build_bias_ext_for_priors(T_crop: int, ext: List[int], blank_id: int,
                               priors_sec: Optional[List[Optional[Tuple[float,float]]]],
                               sec_per_frame: float, crop_start_sec: float,
                               out_penalty: float = PRIOR_OUT_PENALTY,
                               edge_pad_sec: float = PRIOR_EDGE_PAD_SEC) -> Optional[torch.Tensor]:
    """Create bias tensor [T_crop, M] where M=len(ext). For odd indices (token states),
    outside the (start,end) prior window, we add -out_penalty to logprob. Blanks have 0 bias."""
    if not priors_sec:
        return None
    M = len(ext)
    bias = torch.zeros((T_crop, M), dtype=torch.float32)
    # Precompute frame bounds in sec for cropped region
    for i in range((M-1)//2):  # number of tokens
        prior = priors_sec[i] if i < len(priors_sec) else None
        m_idx = 2*i + 1
        if prior is None:
            continue
        start_sec = max(0.0, float(prior[0]) - edge_pad_sec)
        end_sec = max(start_sec, float(prior[1]) + edge_pad_sec)
        # convert to frame indices on cropped timeline
        start_f = int(np.floor((start_sec - crop_start_sec) / max(1e-6, sec_per_frame)))
        end_f = int(np.ceil((end_sec - crop_start_sec) / max(1e-6, sec_per_frame)))
        # mark outside window
        if end_f <= 0 or start_f >= T_crop:
            # no overlap: penalize all frames
            bias[:, m_idx] -= out_penalty
        else:
            if start_f > 0:
                bias[:max(0,start_f), m_idx] -= out_penalty
            if end_f < T_crop:
                bias[max(0,end_f):, m_idx] -= out_penalty
    return bias.to(DEVICE)

def _forced_ctc_jamo_confidences(logprobs: torch.Tensor, ref_ids: List[int], blank_id: int,
                                 confusable_ids: Optional[Dict[int, List[int]]] = None,
                                 bias_ext: Optional[torch.Tensor] = None) -> List[float]:
    """
    Per-ref jamo confidence using Viterbi-local aggregation and non-blank normalization in each local span.
    If bias_ext provided, Viterbi path is biased to prefer frames within the provided time priors.
    """
    ext = _expand_ref_with_blanks(ref_ids, blank_id)  # [B y1 B y2 ... B yN B]
    m_path = _forced_ctc_viterbi_path(logprobs, ext, bias_ext=bias_ext)
    confs: List[float] = []
    for i in range(len(ref_ids)):
        m_idx = 2*i + 1  # position of yi in extended path
        mask = (m_path == m_idx)
        if not mask.any():
            confs.append(0.0); continue
        seg = logprobs[mask]                       # [Ti, V]
        num = seg[:, ref_ids[i]].exp().sum()
        den = (1.0 - seg[:, blank_id].exp()).clamp_min(1e-6).sum()
        ci = float((num / den).item())
        # light confusable bonus within local span
        if confusable_ids and ref_ids[i] in confusable_ids:
            alts = confusable_ids[ref_ids[i]]
            if len(alts) > 0:
                alt_mass = seg[:, alts].exp().sum()
                ci = float(np.clip(ci + 0.5 * float((alt_mass / den).item()), 0.0, 1.0))
        confs.append(ci)
    return confs

# ============ Public API ============
def ctc_char_confidences_from_waveform(waveform: torch.Tensor, sr: int, ref_text_nfd: str,
                                       blank_vad_thresh: float = None, vad_pad_ms: float = None,
                                       temp: float = None, use_confusables: bool = True,
                                       priors_sec: Optional[List[Optional[Tuple[float,float]]]] = None
                                       ) -> List[float]:
    """
    Return per-jamo confidences for NFD Jamo reference (CTC-only).
    If priors_sec is provided, it should be a list aligned with the ref Jamo tokens
    (after filtering by vocab). Each item is (start_sec, end_sec) relative to the
    original waveform timeline. We will crop to VAD slice automatically.
    """
    _init()
    blank_thr = BLANK_VAD_THRESH if blank_vad_thresh is None else blank_vad_thresh
    pad_ms = VAD_PAD_MS if vad_pad_ms is None else vad_pad_ms
    Ttemp = LOGIT_TEMP if temp is None else temp

    wav = _ensure_sr(waveform, sr)
    total_sec = float(wav.shape[-1]) / TARGET_SR
    with torch.inference_mode():
        inputs = _processor(wav.squeeze(0), sampling_rate=TARGET_SR, return_tensors="pt", padding=False)
        input_values = inputs.input_values.to(DEVICE)
        logits = _model(input_values).logits[0]  # [T_full, V]
        logprobs_full = _log_softmax(logits, temp=Ttemp)

        # VAD
        vslice = _blank_posterior_vad(logprobs_full, _blank_id, thresh=blank_thr, pad_ms=pad_ms)
        logprobs = logprobs_full[vslice]  # [T_crop, V]
        T_full = logprobs_full.shape[0]
        T_crop = logprobs.shape[0]
        sec_per_frame = total_sec / max(1, T_full)
        crop_start_sec = vslice.start * sec_per_frame

        # map ref jamo → ids (ignore symbols not in vocab)
        vocab = _processor.tokenizer.get_vocab()
        ref_tokens_all = [t for t in _to_nfd_jamo(ref_text_nfd) if t.strip() != ""]
        ref_ids = []
        aligned_priors: List[Optional[Tuple[float,float]]] = []
        if ref_tokens_all:
            for idx, t in enumerate(ref_tokens_all):
                if t in vocab:
                    ref_ids.append(vocab[t])
                    # align prior list if provided (skip those for OOV tokens)
                    if priors_sec is not None and idx < len(priors_sec):
                        aligned_priors.append(priors_sec[idx])
                    elif priors_sec is not None:
                        aligned_priors.append(None)
        if len(ref_ids) == 0 or logprobs.shape[0] == 0:
            return [0.0] * max(1, len(ref_tokens_all))

        # Build confusable map (Jamo only)
        conf_ids = None
        if use_confusables:
            conf_ids = {}
            for t, alts in CONFUSABLES.items():
                if t in vocab:
                    conf_ids[vocab[t]] = [vocab[a] for a in alts if a in vocab]

        # Build bias for priors
        bias_ext = None
        if aligned_priors and any(p is not None for p in aligned_priors):
            bias_ext = _build_bias_ext_for_priors(
                T_crop, _expand_ref_with_blanks(ref_ids, _blank_id), _blank_id,
                aligned_priors, sec_per_frame, crop_start_sec,
                out_penalty=PRIOR_OUT_PENALTY, edge_pad_sec=PRIOR_EDGE_PAD_SEC
            )

        confs = _forced_ctc_jamo_confidences(logprobs, ref_ids, _blank_id, confusable_ids=conf_ids, bias_ext=bias_ext)

        # pad to original jamo length if some were unknown
        if len(confs) < len(ref_tokens_all):
            confs = confs + [0.0] * (len(ref_tokens_all) - len(confs))
        return confs

def ctc_char_confidences(wav_path: str, ref_text_nfd: str, **kwargs) -> List[float]:
    wav, sr = torchaudio.load(wav_path)
    return ctc_char_confidences_from_waveform(wav, sr, ref_text_nfd, **kwargs)

def _best_path_decode(logprobs: torch.Tensor, vocab_id2tok: Dict[int, str]) -> str:
    ids = logprobs.argmax(dim=-1).tolist()
    out = []
    prev = None
    for i in ids:
        if i == _blank_id:
            prev = i
            continue
        if i != prev:
            out.append(vocab_id2tok.get(i, ""))
        prev = i
    return "".join(out)

def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            insert = current[j-1] + 1
            delete = previous[j] + 1
            replace = previous[j-1] + (ca != cb)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]

def _cer(hyp: str, ref: str) -> float:
    if not ref:
        return 1.0 if hyp else 0.0
    return _levenshtein_distance(hyp, ref) / max(1, len(ref))

def ctc_gate_global_from_waveform(waveform: torch.Tensor, sr: int, ref_text_nfd: str,
                                  cer_threshold: float = 0.65, loss_char_threshold: float = 2.0) -> Dict[str, float]:
    """Global signals for gating (CTC-only, Jamo vocab): CER(best-path jamo vs ref jamo) and mean neglogp."""
    _init()
    wav = _ensure_sr(waveform, sr)
    with torch.inference_mode():
        inputs = _processor(wav.squeeze(0), sampling_rate=TARGET_SR, return_tensors="pt", padding=False)
        input_values = inputs.input_values.to(DEVICE)
        logits = _model(input_values).logits[0]
        logprobs = _log_softmax(logits, temp=LOGIT_TEMP)

        sl = _blank_posterior_vad(logprobs, _blank_id, thresh=BLANK_VAD_THRESH, pad_ms=VAD_PAD_MS)
        logprobs = logprobs[sl]

        vocab = _processor.tokenizer.get_vocab()
        id2tok = {v: k for k, v in vocab.items()}
        hyp = _best_path_decode(logprobs, id2tok)         # jamo string

        print("Hyp:", hyp)

        ref = "".join(_to_nfd_jamo(ref_text_nfd))         # jamo string

        cer = _cer(hyp, ref)

        ref_ids = [vocab[ch] for ch in ref if ch in vocab]
        if len(ref_ids) == 0 or logprobs.shape[0] == 0:
            mean_neglogp = 10.0
        else:
            tok_post = logprobs.exp().mean(0)  # [V]
            pick = tok_post[ref_ids].clamp_min(1e-6)
            mean_neglogp = float((-pick.log()).mean().item())

        return {
            "cer": float(cer),
            "mean_neglogp": mean_neglogp,
            "pass_lenient": (len(ref) <= 2 and cer < 1.0) or (cer < cer_threshold and mean_neglogp < loss_char_threshold),
            "gate_pass": (len(ref) <= 2 and cer < 1.0) or (cer < cer_threshold and mean_neglogp < loss_char_threshold),
        }

def ctc_gate_global(wav_path: str, ref_text_nfd: str, **kwargs) -> Dict[str, float]:
    wav, sr = torchaudio.load(wav_path)
    return ctc_gate_global_from_waveform(wav, sr, ref_text_nfd, **kwargs)