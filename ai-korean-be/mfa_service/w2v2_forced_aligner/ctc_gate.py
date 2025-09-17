# w2v2_forced_aligner/ctc_gate.py
# (JAMO_NFD version) — normalize reference & hypothesis to NFD Jamo (U+1100..U+11FF)
# to match typical Korean CTC vocabularies and stabilize CER/loss on short utterances.

import os
import re
import unicodedata as ud
import logging
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# ===================== Perf & Globals =====================
NUM_THREADS = int(os.getenv("TORCH_NUM_THREADS", "4"))
torch.set_num_threads(NUM_THREADS)
os.environ.setdefault("OMP_NUM_THREADS", str(NUM_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(NUM_THREADS))

TARGET_SR = 16000
MODEL_NAME = os.getenv("W2V_MODEL", "kresnik/wav2vec2-large-xlsr-korean")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fixed choice: JAMO_NFD (드, 니, …)
CTC_TEXT_FORM = "JAMO_NFD"

_log = logging.getLogger(__name__)
if not _log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

_processor: Optional[Wav2Vec2Processor] = None
_ctc_model: Optional[Wav2Vec2ForCTC] = None
_blank_id: Optional[int] = None
_warmed_up: bool = False

_RESAMPLERS: Dict[Tuple[int, int], torchaudio.transforms.Resample] = {}
def _get_resampler(sr_src: int, sr_tgt: int):
    if sr_src == sr_tgt:
        return None
    k = (sr_src, sr_tgt)
    if k not in _RESAMPLERS:
        _RESAMPLERS[k] = torchaudio.transforms.Resample(sr_src, sr_tgt)
    return _RESAMPLERS[k]

# ===================== Text helpers (JAMO_NFD) =====================
# Keep Hangul syllables + modern Jamo + compat Jamo + space (for stripping)
_HANGUL_KEEP = re.compile(r"[^\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F\s]")

def _strip_keep_hangul(text: str) -> str:
    t = text.strip()
    t = _HANGUL_KEEP.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _to_jamo_nfd_no_space(text: str) -> str:
    """
    Normalize to NFD Jamo (U+1100..U+11FF), remove spaces.
    Example: '드 니' -> '드니'
    """
    t = _strip_keep_hangul(text)
    t = ud.normalize("NFD", t)
    return t.replace(" ", "")

def _normalize_hyp_to_jamo_nfd(text: str) -> str:
    """
    Normalize hypothesis to NFD Jamo, keep only Hangul/Jamo, remove spaces.
    """
    t = ud.normalize("NFD", text)
    t = _strip_keep_hangul(t)
    return t.replace(" ", "")

def _encode_ctc_ids(text: str) -> Tuple[List[int], str]:
    """
    Encode text in NFD Jamo (no spaces) for CTC. Returns (ids, ref_text_used).
    """
    assert _processor is not None
    t2 = _to_jamo_nfd_no_space(text)
    if not t2:
        return [], ""
    ids = _processor.tokenizer(t2, add_special_tokens=False)["input_ids"]
    ids = [i for i in ids if i is not None]
    return ids, t2

# ===================== Init =====================
def init_ctc() -> None:
    global _processor, _ctc_model, _blank_id, _warmed_up
    if _processor is None:
        _processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    if _ctc_model is None:
        _ctc_model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(DEVICE).eval()

    # Determine blank id robustly: prefer config.*blank*, then pad_token_id, else 0
    bid = None
    for key in ("blank_token_id", "ctc_blank_token_id", "pad_token_id"):
        val = getattr(_ctc_model.config, key, None)
        if isinstance(val, int) and val >= 0:
            bid = val
            break
    if bid is None:
        tok = getattr(_processor, "tokenizer", None)
        if tok is not None:
            bid = getattr(tok, "pad_token_id", None)
    _blank_id = int(bid) if isinstance(bid, int) and bid >= 0 else 0

    if not _warmed_up:
        try:
            with torch.inference_mode():
                dummy = torch.zeros(1, TARGET_SR, dtype=torch.float32, device=DEVICE)
                _ = _ctc_model(
                    _processor(dummy, sampling_rate=TARGET_SR, return_tensors="pt").input_values.to(DEVICE)
                )
            _warmed_up = True
            _log.info(
                "CTC warmup done for %s on %s (blank_id=%s, threads=%d)",
                MODEL_NAME, DEVICE, _blank_id, NUM_THREADS
            )
        except Exception as e:
            _warmed_up = True
            _log.warning("CTC warmup failed (continue): %s", e)

# ===================== Light VAD (less aggressive) =====================
def _light_vad(mono: torch.Tensor, sr: int, pad_ms: float = 80.0) -> torch.Tensor:
    """
    Trim silence very lightly to avoid chopping short onsets/codas.
    """
    x = mono.detach().cpu().float().numpy()
    if x.size == 0:
        return mono
    frame = max(1, int(round(20 * sr / 1000.0)))
    hop = max(1, int(round(10 * sr / 1000.0)))
    frames = [x[i:i + frame] for i in range(0, max(1, len(x) - frame + 1), hop)] or [x]
    rms = np.sqrt(np.mean(np.stack(frames, 0) ** 2, axis=1))
    thr = np.quantile(rms, 0.10)  # lenient
    idx = np.where(rms > thr)[0]
    if idx.size == 0:
        return mono
    i0, i1 = int(idx[0]), int(idx[-1]) + 1
    pad = max(1, int(round(pad_ms * sr / 1000.0 / hop)))
    s = max(0, (i0 - pad) * hop)
    e = min(len(x), (i1 + pad) * hop)
    y = torch.from_numpy(x[s:e]).to(mono.dtype)
    peak = float(y.abs().max().item() or 1.0)
    if peak > 0:
        y = (y / peak).clamp_(-1.0, 1.0)
    return y

# ===================== Core APIs =====================
@torch.inference_mode()
def ctc_gate_global(wav_path: str, reference_text: str) -> Dict[str, float]:
    init_ctc()
    wav, sr = torchaudio.load(wav_path)
    if wav.ndim == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        rs = _get_resampler(sr, TARGET_SR)
        wav = rs(wav) if rs is not None else wav
        sr = TARGET_SR
    return ctc_gate_global_from_waveform(wav.squeeze(0), sr, reference_text)

@torch.inference_mode()
def ctc_gate_global_from_waveform(x: torch.Tensor, sr: int, reference_text: str) -> Dict[str, float]:
    """
    Returns:
      - loss_per_char: average CTC loss per target char (ids length)
      - cer: CER on id sequence (bounded in [0,1])
      - cer_text: CER on decoded text normalized to NFD Jamo (bounded in [0,1])
    """
    init_ctc()

    mono = x.squeeze(0) if x.ndim == 2 else x
    mono = mono.to(torch.float32).clamp_(-1, 1)
    if sr != TARGET_SR:
        rs = _get_resampler(sr, TARGET_SR)
        mono = rs(mono.unsqueeze(0)).squeeze(0) if rs is not None else mono
    mono = _light_vad(mono, TARGET_SR)

    # Short/silent safeguard + AUTO-GAIN for tiny inputs
    dur_sec = float(mono.numel()) / TARGET_SR if mono.numel() else 0.0
    rms = float(mono.pow(2).mean().sqrt().item()) if mono.numel() else 0.0

    # nếu RMS quá nhỏ, khuếch đại nhẹ để model "nghe thấy" (giới hạn 10x)
    if 0.0 < rms < 0.012:
        target_rms = 0.06
        gain = min(12.0, target_rms / max(rms, 1e-6))
        mono = (mono * gain).clamp_(-1.0, 1.0)
        rms = float(mono.pow(2).mean().sqrt().item())

    # Quá ngắn/yếu -> coi như off-text (để tầng trên hard-reject).
    if dur_sec < 0.12 or rms < 1e-4:
        return {"loss_per_char": 99.0, "cer": 1.0, "cer_text": 1.0}

    inputs = _processor(mono, sampling_rate=TARGET_SR, return_tensors="pt").input_values.to(DEVICE)
    logits = _ctc_model(inputs).logits                     # [1, T, V]
    log_probs = torch.log_softmax(logits, dim=-1)          # [1, T, V]
    T = log_probs.shape[1]

    # Encode reference in JAMO_NFD (no spaces)
    labels, ref_text_used = _encode_ctc_ids(reference_text)
    if len(labels) == 0:
        return {"loss_per_char": 99.0, "cer": 1.0, "cer_text": 1.0}

    input_lengths = torch.tensor([T], dtype=torch.long, device=log_probs.device)
    target = torch.tensor([labels], dtype=torch.long, device=log_probs.device)
    target_lengths = torch.tensor([len(labels)], dtype=torch.long, device=log_probs.device)

    loss = torch.nn.functional.ctc_loss(
        log_probs.transpose(0, 1),  # [T, B, V]
        target,
        input_lengths,
        target_lengths,
        blank=_blank_id or 0,
        reduction="mean",
        zero_infinity=True,
    )
    loss_per_char = float(loss.item() / max(1, len(labels)))

    # ==== CER id-level (bounded) ====
    blank = _blank_id or 0
    pred_ids = log_probs.argmax(dim=-1)[0].detach().cpu().tolist()
    collapsed, prev = [], None
    for t in pred_ids:
        if t == blank:
            prev = None
            continue
        if t != prev:
            collapsed.append(t)
            prev = t

    dp = np.zeros((len(collapsed) + 1, len(labels) + 1), dtype=np.int32)
    dp[:, 0] = np.arange(len(collapsed) + 1)
    dp[0, :] = np.arange(len(labels) + 1)
    for i in range(1, len(collapsed) + 1):
        ai = collapsed[i - 1]
        for j in range(1, len(labels) + 1):
            c = 0 if ai == labels[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + c)
    cer_id = float(dp[-1, -1]) / max(1, len(labels), len(collapsed))
    cer_id = max(0.0, min(1.0, cer_id))

    # ==== CER text-level on NFD Jamo (bounded) ====
    def _decode_text(lprobs: torch.Tensor, blank_id: int) -> str:
        ids = lprobs.argmax(dim=-1)[0].detach().cpu().tolist()
        out, prev = [], None
        for t in ids:
            if t == blank_id:
                prev = None
                continue
            if t != prev:
                out.append(t)
                prev = t
        try:
            return _processor.tokenizer.decode(out, skip_special_tokens=True)
        except Exception:
            return "".join(map(str, out))

    hyp_decoded = _decode_text(log_probs, blank)
    print("hyp_decoded:", repr(hyp_decoded), "|| ref_text_used:", repr(ref_text_used))
    hyp_norm = _normalize_hyp_to_jamo_nfd(hyp_decoded)
    print("hyp_norm:", repr(hyp_norm), "|| ref_text_used:", repr(ref_text_used))
    ref_norm = ref_text_used  # already Jamo NFD without spaces

    dp2 = np.zeros((len(hyp_norm) + 1, len(ref_norm) + 1), dtype=np.int32)
    dp2[:, 0] = np.arange(len(hyp_norm) + 1)
    dp2[0, :] = np.arange(len(ref_norm) + 1)
    for i in range(1, len(hyp_norm) + 1):
        for j in range(1, len(ref_norm) + 1):
            c = 0 if hyp_norm[i - 1] == ref_norm[j - 1] else 1
            dp2[i, j] = min(dp2[i - 1, j] + 1, dp2[i, j - 1] + 1, dp2[i - 1, j - 1] + c)
    cer_text = float(dp2[-1, -1]) / max(1, len(ref_norm), len(hyp_norm))
    cer_text = max(0.0, min(1.0, cer_text))

    return {
        "loss_per_char": loss_per_char,
        "cer": float(cer_id),
        "cer_text": float(cer_text),
    }

@torch.inference_mode()
def ctc_char_confidences(wav_path: str, reference_text: str) -> List[float]:
    init_ctc()
    wav, sr = torchaudio.load(wav_path)
    if wav.ndim == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        rs = _get_resampler(sr, TARGET_SR)
        wav = rs(wav) if rs is not None else wav
        sr = TARGET_SR
    return ctc_char_confidences_from_waveform(wav.squeeze(0), sr, reference_text)

@torch.inference_mode()
def ctc_char_confidences_from_waveform(x: torch.Tensor, sr: int, reference_text: str) -> List[float]:
    """
    Compute per-char posteriors for the Jamo-NFD reference sequence using
    forward-backward on the extended target. Map back to reference chars.
    """
    init_ctc()

    mono = x.squeeze(0) if x.ndim == 2 else x
    mono = mono.to(torch.float32).clamp_(-1, 1)
    if sr != TARGET_SR:
        rs = _get_resampler(sr, TARGET_SR)
        mono = rs(mono.unsqueeze(0)).squeeze(0) if rs is not None else mono
    mono = _light_vad(mono, TARGET_SR)
    if mono.numel() == 0:
        return [0.0] * max(1, len(reference_text))

    inputs = _processor(mono, sampling_rate=TARGET_SR, return_tensors="pt").input_values.to(DEVICE)
    logits = _ctc_model(inputs).logits
    log_probs = torch.log_softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()  # [T, V]
    T, V = log_probs.shape

    labels, ref_used = _encode_ctc_ids(reference_text)
    if not labels:
        return [0.0] * max(1, len(reference_text))
    blank = _blank_id or 0

    # Build extended target: b y1 b y2 b ... yU b
    ext = [blank]
    for y in labels:
        ext += [y, blank]
    L = len(ext)

    # Forward
    alpha = np.full((T, L), -np.inf, dtype=np.float32)
    alpha[0, 0] = log_probs[0, ext[0]]
    if L > 1:
        alpha[0, 1] = log_probs[0, ext[1]]
    for t in range(1, T):
        for s in range(L):
            stay = alpha[t - 1, s]
            move = alpha[t - 1, s - 1] if s - 1 >= 0 else -np.inf
            skip = -np.inf
            if s - 2 >= 0 and not (ext[s] == blank or ext[s] == ext[s - 2]):
                skip = alpha[t - 1, s - 2]
            m = max(stay, move, skip)
            alpha[t, s] = m + log_probs[t, ext[s]]

    # Backward
    beta = np.full((T, L), -np.inf, dtype=np.float32)
    beta[T - 1, L - 1] = log_probs[T - 1, ext[L - 1]]
    if L > 1:
        beta[T - 1, L - 2] = log_probs[T - 1, ext[L - 2]]
    for t in range(T - 2, -1, -1):
        for s in range(L):
            stay = beta[t + 1, s]
            move = beta[t + 1, s + 1] if s + 1 < L else -np.inf
            skip = -np.inf
            if s + 2 < L and not (ext[s] == blank or ext[s] == ext[s + 2]):
                skip = beta[t + 1, s + 2]
            m = max(stay, move, skip)
            beta[t, s] = m + log_probs[t, ext[s]]

    # Posterior
    logZ = np.logaddexp(alpha[T - 1, L - 1], alpha[T - 1, L - 2] if L - 2 >= 0 else -np.inf)
    post = alpha + beta - logZ
    post = np.exp(np.clip(post, -30, 30))  # numeric stability

    U = len(labels)
    conf = np.zeros(U, dtype=np.float32)
    for u in range(U):
        s = 2 * u + 1  # position of label y_u in extended sequence
        if s < L:
            conf[u] = float(post[:, s].mean())
    if conf.sum() > 1e-8:
        conf = conf / conf.max()
    conf = np.clip(conf, 0.0, 1.0)

    # Map confidences to the Jamo-NFD reference string (ref_used)
    # One confidence per reference jamo char.
    out = []
    i = 0
    for _c in ref_used:
        # ref_used is already Jamo NFD, 1 char ↔ 1 label approximately.
        if i < U:
            out.append(float(conf[i] ** 0.7))
            i += 1
        else:
            out.append(0.6)  # fallback if mismatch (should be rare)
    return out
