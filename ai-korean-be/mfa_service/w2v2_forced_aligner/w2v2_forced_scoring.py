# w2v2_forced_aligner/w2v2_forced_scoring.py
import os
import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from textgrid import TextGrid

from utils.utils_advice import advices_for_phoneme, VOWELS, STOP_UNRELEASED

# (NEW) — CTC tools moved out to ctc_gate.py
# Adjust the import path to your project layout if needed.
from w2v2_forced_aligner.ctc_gate import (
    ctc_gate_global,
    ctc_gate_global_from_waveform,
    ctc_char_confidences,
    ctc_char_confidences_from_waveform,
)

# ============ Perf knobs ============
NUM_THREADS = int(os.getenv("TORCH_NUM_THREADS", "4"))
torch.set_num_threads(NUM_THREADS)
os.environ.setdefault("OMP_NUM_THREADS", str(NUM_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(NUM_THREADS))

TARGET_SR = 16000
REF_CACHE_MAX = int(os.getenv("REF_CACHE_MAX", "128"))

_log = logging.getLogger(__name__)
if not _log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

# ============ 1) Model & device ============
MODEL_NAME = os.getenv("W2V_MODEL", "kresnik/wav2vec2-large-xlsr-korean")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Lazy globals (init ở startup)
_processor: Optional[Wav2Vec2Processor] = None
_model: Optional[Wav2Vec2Model] = None
_warmed_up: bool = False

def init_w2v2() -> None:
    """Load & warmup Wav2Vec2 (idempotent)."""
    global _processor, _model, _warmed_up
    if _processor is None:
        _processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    if _model is None:
        _model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    if not _warmed_up:
        try:
            with torch.inference_mode():
                dummy = torch.zeros(1, TARGET_SR, dtype=torch.float32, device=DEVICE)
                _ = _model(dummy)
            _warmed_up = True
            _log.info("W2V2 warmup done for %s on %s (threads=%d)", MODEL_NAME, DEVICE, NUM_THREADS)
        except Exception as e:
            _log.warning("Warmup failed (safe to ignore): %s", e)
            _warmed_up = True

# ============ Helpers ============
def _is_vowel(ph: str) -> bool:
    return ph in VOWELS

def _estimate_position(curr: Optional[str], prev_: Optional[str], next_: Optional[str]) -> Optional[str]:
    if not curr:
        return None
    if curr in STOP_UNRELEASED:
        return "coda"
    if _is_vowel(curr):
        return "nucleus"
    if _is_vowel(next_ or ""):
        return "onset"
    return "coda"

_RESAMPLERS: Dict[Tuple[int, int], torchaudio.transforms.Resample] = {}
def _get_resampler(orig_sr: int, new_sr: int):
    if orig_sr == new_sr:
        return None
    key = (orig_sr, new_sr)
    if key not in _RESAMPLERS:
        _RESAMPLERS[key] = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=new_sr)
    return _RESAMPLERS[key]

ALIAS_CANON: Dict[str, str] = {
    "tɕ": "t͡ɕ", "tɕ͈": "t͡ɕ͈", "tɕʰ": "t͡ɕʰ",
    "e̞": "e", "o̞": "o", "ɐ": "a", "ɫ": "l",
    "ɾ̚": "ɾ",
}
def _canon(ph: str) -> str: return ALIAS_CANON.get(ph, ph)
def _canon_seq(seq: List[str]) -> List[str]: return [_canon(x) for x in seq]

# --- Waveform normalization helpers ---
def _vad_trim(mono: torch.Tensor, sr: int, frame_ms: float = 20.0, hop_ms: float = 10.0,
              q_lo: float = 0.15, pad_ms: float = 150.0) -> torch.Tensor:
    """Trim im lặng 2 đầu bằng RMS quantile; vẫn an toàn nếu toàn im lặng."""
    x = mono.detach().cpu().float().numpy()
    if x.size == 0:
        return mono
    frame = max(1, int(round(frame_ms * sr / 1000.0)))
    hop = max(1, int(round(hop_ms * sr / 1000.0)))
    frames = [x[i:i+frame] for i in range(0, max(1, len(x)-frame+1), hop)] or [x]
    rms = np.sqrt(np.mean(np.stack(frames, 0)**2, axis=1))
    thr = np.quantile(rms, q_lo)
    idx = np.where(rms > thr)[0]
    if idx.size == 0:
        return mono
    i0, i1 = int(idx[0]), int(idx[-1]) + 1
    pad = max(1, int(round(pad_ms * sr / 1000.0 / hop)))
    s = max(0, (i0 - pad) * hop)
    e = min(len(x), (i1 + pad) * hop)
    y = torch.from_numpy(x[s:e]).to(mono.dtype)
    # normalize gain nhẹ
    peak = float(y.abs().max().item() or 1.0)
    if peak > 0:
        y = (y / peak).clamp_(-1.0, 1.0)
    return y

def _only_hangul(s: str) -> str:
    return "".join(ch for ch in s if 0xAC00 <= ord(ch) <= 0xD7A3)

def _normalize_waveform_tensor(waveform: torch.Tensor, sr: int, target_sr: int = TARGET_SR) -> torch.Tensor:
    """
    waveform: [C,N] hoặc [N]; trả về mono float32 [N] CPU, resample -> target_sr
    """
    if waveform is None or waveform.numel() == 0:
        return torch.zeros(0, dtype=torch.float32)
    x = waveform
    if x.ndim == 2 and x.shape[0] > 1:
        x = x.mean(dim=0, keepdim=False)
    elif x.ndim == 2 and x.shape[0] == 1:
        x = x.squeeze(0)
    x = x.to(torch.float32).clamp_(-1.0, 1.0)
    if sr != target_sr:
        rs = _get_resampler(sr, target_sr)
        x = rs(x.unsqueeze(0)).squeeze(0) if rs is not None else x
    return x.cpu()

# ============ 2) Feature extraction ============
@torch.inference_mode()
def extract_features(wav_path: str) -> Tuple[torch.Tensor, int, float]:
    init_w2v2()
    waveform, sr = torchaudio.load(wav_path)  # [C,N]
    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        resampler = _get_resampler(sr, TARGET_SR)
        waveform = resampler(waveform) if resampler is not None else waveform
        sr = TARGET_SR
    mono = waveform.squeeze(0)
    inputs = _processor(mono, sampling_rate=sr, return_tensors="pt").input_values.to(DEVICE)
    outputs = _model(inputs)
    feats = outputs.last_hidden_state.squeeze(0).detach().cpu()  # [T,D]
    total_audio_sec = float(mono.shape[0]) / float(sr)
    return feats, sr, total_audio_sec

@torch.inference_mode()
def extract_features_from_waveform(waveform: torch.Tensor, sr: int) -> Tuple[torch.Tensor, int, float]:
    """Phiên bản không I/O: nhận waveform tensor."""
    init_w2v2()
    mono = _normalize_waveform_tensor(waveform, sr, TARGET_SR)  # [N]
    if mono.numel() == 0:
        return torch.zeros(1,1), TARGET_SR, 1e-3
    inputs = _processor(mono, sampling_rate=TARGET_SR, return_tensors="pt").input_values.to(DEVICE)
    outputs = _model(inputs)
    feats = outputs.last_hidden_state.squeeze(0).detach().cpu()
    total_audio_sec = float(mono.shape[0]) / float(TARGET_SR)
    return feats, TARGET_SR, total_audio_sec

# ========= 2b) Char confidences từ năng lượng =========
def estimate_char_confidences_from_energy(
    wav_path: str, reference_text: str, *, sr_target: int = TARGET_SR,
    frame_ms: float = 20.0, hop_ms: float = 10.0, vad_lo_q: float = 0.15,
    soft_q_lo: float = 0.18, soft_q_hi: float = 0.82, lead_pad_ms: float = 200.0,
    tail_pad_ms: float = 200.0, edge_comp: float = 1.0, smooth_win: int = 7,
) -> List[float]:
    wav, sr = torchaudio.load(wav_path)
    if wav.ndim == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sr_target:
        resampler = _get_resampler(sr, sr_target)
        wav = resampler(wav) if resampler is not None else wav
        sr = sr_target
    return _energy_conf_from_mono(wav.squeeze(0).numpy().astype(np.float32), sr, reference_text,
                                  frame_ms, hop_ms, vad_lo_q, soft_q_lo, soft_q_hi,
                                  lead_pad_ms, tail_pad_ms, edge_comp, smooth_win)

def estimate_char_confidences_from_energy_from_waveform(
    waveform: torch.Tensor, sr: int, reference_text: str, *,
    frame_ms: float = 20.0, hop_ms: float = 10.0, vad_lo_q: float = 0.15,
    soft_q_lo: float = 0.2, soft_q_hi: float = 0.8, lead_pad_ms: float = 200.0,
    tail_pad_ms: float = 200.0, edge_comp: float = 1.0, smooth_win: int = 9,
) -> List[float]:
    mono = _normalize_waveform_tensor(waveform, sr, TARGET_SR).numpy().astype(np.float32)
    return _energy_conf_from_mono(mono, TARGET_SR, reference_text,
                                  frame_ms, hop_ms, vad_lo_q, soft_q_lo, soft_q_hi,
                                  lead_pad_ms, tail_pad_ms, edge_comp, smooth_win)

def _energy_conf_from_mono(
    x: np.ndarray, sr: int, reference_text: str,
    frame_ms: float, hop_ms: float, vad_lo_q: float,
    soft_q_lo: float, soft_q_hi: float, lead_pad_ms: float,
    tail_pad_ms: float, edge_comp: float, smooth_win: int,
) -> List[float]:
    Nchar = max(1, len(reference_text))
    if x.size == 0:
        return [0.0] * Nchar

    frame = max(1, int(round(frame_ms * sr / 1000.0)))
    hop = max(1, int(round(hop_ms * sr / 1000.0)))
    frames = [x[i:i + frame] for i in range(0, max(1, len(x) - frame + 1), hop)] or [x]
    mat = np.stack(frames, axis=0)
    rms = np.sqrt(np.mean(mat * mat, axis=1))
    log_rms = 20.0 * np.log10(np.maximum(rms, 1e-8))

    thr_vad = np.quantile(log_rms, vad_lo_q)
    idx = np.where(log_rms > thr_vad)[0]
    if idx.size == 0:
        return [0.0] * Nchar

    i0, i1 = int(idx[0]), int(idx[-1]) + 1
    lead = max(0, int(round(lead_pad_ms * sr / 1000.0 / hop)))
    tail = max(0, int(round(tail_pad_ms * sr / 1000.0 / hop)))
    s = max(0, i0 - lead)
    e = min(len(log_rms), i1 + tail)
    log_rms = log_rms[s:e]

    # === NEW: nếu số frame sau VAD ít hơn số ký tự → nội suy thay vì trả 0 ===
    if log_rms.size < Nchar:
        if log_rms.size == 0:
            return [0.0] * Nchar
        idx_src = np.arange(len(log_rms), dtype=np.float32)
        idx_tgt = np.linspace(0.0, len(log_rms) - 1.0, num=Nchar, dtype=np.float32)
        log_rms = np.interp(idx_tgt, idx_src, log_rms)

    lo = np.quantile(log_rms, 0.16)
    hi = np.quantile(log_rms, 0.84)
    denom = max(1e-6, (hi - lo))
    voiced = np.clip((log_rms - lo) / denom, 0.0, 1.0).astype(np.float32)

    F = len(voiced)
    per = F / float(Nchar)
    bounds = [0] + [int(round(k * per)) for k in range(1, Nchar)] + [F]
    confs = [float(voiced[a:b].mean()) if b > a else 0.0 for a, b in zip(bounds[:-1], bounds[1:])]

    if Nchar >= 1: confs[0] *= edge_comp
    if Nchar >= 2: confs[-1] *= edge_comp

    if smooth_win > 1 and Nchar > 2:
        k = np.ones(smooth_win, dtype=np.float32) / smooth_win
        confs = list(np.convolve(np.array(confs, dtype=np.float32), k, mode="same"))

    confs = np.asarray(confs, dtype=np.float32)
    confs = np.clip(confs, 0.0, 1.0) ** 0.5

    m = float(confs.mean()) if confs.size else 0.0
    beta = float(np.clip((0.68 - m) / 0.20, 0.0, 0.65))
    if beta > 0.0: confs = (1.0 - beta) * confs + beta * 0.75
    if confs.size >= 1: confs[0] = min(1.0, confs[0] + 0.07)
    if confs.size >= 2: confs[1] = min(1.0, confs[1] + 0.05)

    return np.clip(confs, 0.0, 1.0).astype(np.float32).tolist()

# ============ 3) Load TextGrid (phones) ============
def _pick_phone_tier(tg: TextGrid):
    candidates = {"phones", "phoneme", "phone", "PHONES", "Phones"}
    for tier in tg.tiers:
        if getattr(tier, "name", None) in candidates:
            return tier
    for tier in tg.tiers:
        try:
            _ = tier[0]
            return tier
        except Exception:
            continue
    return None

_DROP_LABELS = {"", "sp", "sil", "spn", "pau", "ns", "breath", "noise"}

def load_alignment(textgrid_path: str) -> List[Dict[str, Any]]:
    tg = TextGrid(); tg.read(textgrid_path)
    tier = _pick_phone_tier(tg)
    if tier is None:
        return []
    phones = []
    for itv in tier:
        mark = (itv.mark or "").strip()
        if not mark or mark.lower() in _DROP_LABELS:
            continue
        phones.append({
            "phoneme": mark,
            "start": float(itv.minTime),
            "end": float(itv.maxTime),
        })
    return phones

# ============ 4) Map time → phoneme vectors ============
def extract_phoneme_vectors(
    features: torch.Tensor, sr: int, phoneme_times: List[Dict[str, Any]], total_audio_sec: float
) -> List[Dict[str, Any]]:
    T = features.shape[0]
    phoneme_vectors: List[Dict[str, Any]] = []
    denom = max(total_audio_sec, 1e-8)
    for p in phoneme_times:
        start_idx = int((p["start"] / denom) * T)
        end_idx   = int((p["end"]   / denom) * T)
        start_idx = max(0, min(start_idx, T - 1))
        end_idx   = max(0, min(end_idx,   T))
        if end_idx <= start_idx:
            end_idx = min(start_idx + 1, T)
        seg = features[start_idx:end_idx]
        vec = features[start_idx] if seg.numel() == 0 else seg.mean(dim=0)
        phoneme_vectors.append({
            "phoneme": p["phoneme"],
            "vector": vec.detach().float(),
            "start": p["start"],
            "end": p["end"],
        })
    return phoneme_vectors

# ============ 5) Cosine score ============
def cosine_score_torch(v1: torch.Tensor, v2: torch.Tensor) -> float:
    v1 = v1 / (v1.norm(p=2) + 1e-9)
    v2 = v2 / (v2.norm(p=2) + 1e-9)
    sim = float((v1 * v2).sum().item())
    return max(0.0, min(1.0, sim))

# ============ 6) Needleman–Wunsch on labels ============
def _nw_align_labels(
    ref_labels: List[str], hyp_labels: List[str],
    match_score: float = 1.0, mismatch_penalty: float = -0.5, gap_penalty: float = -0.6,
) -> List[Tuple[Optional[int], Optional[int]]]:
    n, m = len(ref_labels), len(hyp_labels)
    dp = [[0.0]*(m+1) for _ in range(n+1)]
    bt = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        dp[i][0] = dp[i-1][0] + gap_penalty; bt[i][0] = 1
    for j in range(1, m+1):
        dp[0][j] = dp[0][j-1] + gap_penalty; bt[0][j] = 2
    for i in range(1, n+1):
        for j in range(1, m+1):
            d = dp[i-1][j-1] + (match_score if ref_labels[i-1] == hyp_labels[j-1] else mismatch_penalty)
            u = dp[i-1][j]   + gap_penalty
            l = dp[i][j-1]   + gap_penalty
            if d >= u and d >= l: dp[i][j] = d; bt[i][j] = 0
            elif u >= l:        dp[i][j] = u; bt[i][j] = 1
            else:               dp[i][j] = l; bt[i][j] = 2
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
    return pairs

# ============ 7) Chunk label mapping ============
def _build_chunk_labels_for_ref(
    vectors_ref: List[Dict[str, Any]], ref_chunks: Optional[List[Tuple[str, List[str]]]],
) -> List[str]:
    N = len(vectors_ref)
    if not ref_chunks or N == 0:
        return [""] * N
    expect_counts = []
    for label, phs in ref_chunks:
        c = sum(1 for p in phs if p and p != "Ø")
        expect_counts.append(max(1, c))
    tot_expect = sum(expect_counts)
    if tot_expect == N:
        out, idx = [], 0
        for (label, _), cnt in zip(ref_chunks, expect_counts):
            for _ in range(cnt):
                if idx < N: out.append(label); idx += 1
        while len(out) < N: out.append(ref_chunks[-1][0])
        return out
    out, idx = [], 0
    for (label, _), cnt in zip(ref_chunks, expect_counts):
        alloc = int(round((cnt / max(1, tot_expect)) * N))
        rem_chunks = len(ref_chunks) - len(out) - 1
        rem_slots = N - idx
        if rem_chunks <= 0: alloc = rem_slots
        else: alloc = min(alloc, rem_slots - rem_chunks)
        alloc = max(1, alloc)
        for _ in range(alloc):
            if idx < N: out.append(label); idx += 1
    while len(out) < N: out.append(ref_chunks[-1][0])
    return out

# ============ 8A) Scoring + Advice (MFA-W2V; cần TextGrid user) ============
def score_pronunciation(
    vectors_user: List[Dict[str, Any]], vectors_ref: List[Dict[str, Any]], phones_user: List[Dict[str, Any]],
    *, ref_chunks: Optional[List[Tuple[str, List[str]]]] = None, advice_threshold: float = 0.8,
) -> Dict[str, Any]:
    ref_labels_orig = [d["phoneme"] for d in vectors_ref]
    hyp_labels_orig = [d["phoneme"] for d in vectors_user]
    ref_labels = _canon_seq(ref_labels_orig)
    hyp_labels = _canon_seq(hyp_labels_orig)
    pairs = _nw_align_labels(ref_labels, hyp_labels)
    ref2hyp: Dict[int, Optional[int]] = {}
    for (ri, hj) in pairs:
        if ri is not None: ref2hyp[ri] = hj
    details: List[Dict[str, Any]] = []
    for i, (r_lab, r_raw) in enumerate(zip(ref_labels, ref_labels_orig)):
        hj = ref2hyp.get(i, None); missing = False
        if hj is not None and hj < len(vectors_user):
            sim = cosine_score_torch(vectors_user[hj]["vector"], vectors_ref[i]["vector"])
            matched_label = hyp_labels[hj]
        else:
            sim = 0.0; matched_label = None; missing = True
        color = "green" if sim >= 0.8 else ("yellow" if sim >= 0.6 else "red")
        prev_ref = ref_labels[i - 1] if i > 0 else None
        next_ref = ref_labels[i + 1] if i + 1 < len(ref_labels) else None
        position = _estimate_position(r_lab, prev_ref, next_ref)
        env_hint = None
        if matched_label is not None and matched_label != r_lab:
            if ("͈" in r_lab) != ("͈" in matched_label): env_hint = "tense_diff"
            elif (r_lab.endswith("ʰ")) != (matched_label.endswith("ʰ")): env_hint = "asp_diff"
            elif (r_lab in {"k̚","t̚","p̚"}) != (matched_label in {"k̚","t̚","p̚"}): env_hint = "release_diff"
            elif (r_lab in {"j","w"}) != (matched_label in {"j","w"}): env_hint = "glide_diff"
        tips = advices_for_phoneme(curr=r_lab, prev=prev_ref, next_=next_ref,
                                   position=position, low_score=(sim < advice_threshold), env_hint=env_hint)
        if missing: tips = ["Âm này bị thiếu (bỏ/nuốt âm hoặc không khớp)."] + tips
        display_label = r_raw if position == "coda" and r_raw in STOP_UNRELEASED else r_lab
        details.append({
            "phoneme": display_label, "phoneme_surface": r_lab, "score": round(float(sim), 4),
            "start": float(vectors_ref[i].get("start")) if vectors_ref[i].get("start") is not None else None,
            "end": float(vectors_ref[i].get("end")) if vectors_ref[i].get("end") is not None else None,
            "color": color, "advice": tips, "position": position,
        })
    avg = float(np.mean([d["score"] for d in details])) if details else 0.0
    by_chunk: List[Dict[str, Any]] = []
    if ref_chunks:
        chunk_labels_seq = _build_chunk_labels_for_ref(vectors_ref, ref_chunks)
        for d, lab in zip(details, chunk_labels_seq): d["label"] = lab
        cur = None; cur_ph, cur_sc, cur_adv = [], [], []
        for d in details:
            lab = d.get("label", "")
            if cur is None: cur = lab
            if lab != cur:
                if cur_ph:
                    avg_c = float(np.mean(cur_sc)) if cur_sc else 0.0
                    by_chunk.append({"label": cur, "phonemes": cur_ph, "scores": cur_sc,
                                     "avg_score": avg_c, "advice": list(dict.fromkeys(cur_adv))[:3]})
                cur = lab; cur_ph, cur_sc, cur_adv = [], [], []
            cur_ph.append(d["phoneme"]); cur_sc.append(d["score"]); cur_adv.extend(d.get("advice") or [])
        if cur_ph:
            avg_c = float(np.mean(cur_sc)) if cur_sc else 0.0
            by_chunk.append({"label": cur or "", "phonemes": cur_ph, "scores": cur_sc,
                             "avg_score": avg_c, "advice": list(dict.fromkeys(cur_adv))[:3]})
    else:
        for d in details: d["label"] = ""
    return {"avg_score": avg, "details": details, "details_collapsed": by_chunk, "pairs": pairs}

# ============ 8C) Scoring (Forced) + (external) CTC gate ============
def _build_duration_priors_from_ref_tg(
    ref_textgrid_path: str, ref_chunks: List[Tuple[str, List[str]]], *,
    phone_tier: str = "phones", drop_labels: Optional[set] = None,
) -> List[List[float]]:
    drop_labels = drop_labels or {"", "sil", "sp", "spn", "pau", "noise", "breath"}
    tg = TextGrid(); tg.read(ref_textgrid_path)
    tier = None
    for t in tg.tiers:
        if getattr(t, "name", "").lower() == phone_tier.lower():
            tier = t; break
    if tier is None:
        for t in tg.tiers:
            try:
                _ = t[0]; tier = t; break
            except Exception:
                continue
    if tier is None:
        return [[] for _ in ref_chunks]

    tg_phones, tg_durs = [], []
    for itv in tier:
        lab = (itv.mark or "").strip()
        if lab in drop_labels: continue
        tg_phones.append(_canon(lab))
        tg_durs.append(float(itv.maxTime) - float(itv.minTime))

    g2p_flat, spans = [], []
    cur = 0
    for _, phs in ref_chunks:
        phs_c = [_canon(p) for p in phs]
        g2p_flat.extend(phs_c)
        spans.append((cur, cur + len(phs_c)))
        cur += len(phs_c)

    pairs = _nw_align_labels(g2p_flat, tg_phones)
    dur_per_g2p = [0.0] * len(g2p_flat)
    for gi, tj in pairs:
        if gi is not None and tj is not None:
            dur_per_g2p[gi] += tg_durs[tj]

    priors: List[List[float]] = []
    for (s, e), (_, phs) in zip(spans, ref_chunks):
        lens = e - s
        if lens <= 0: priors.append([]); continue
        local = dur_per_g2p[s:e]; total = sum(local)
        priors.append([1.0 / lens] * lens if total <= 1e-8 else [x / total for x in local])
    return priors

def _default_phone_weights(phones: List[str]) -> List[float]:
    if not phones: return []
    vals = []
    for ph in phones:
        if _is_vowel(ph): vals.append(1.5)
        elif ph in {"j", "w"}: vals.append(1.1)
        else: vals.append(1.0)
    s = sum(vals)
    return [v / s for v in vals]

def _ctc_lenient_gate(ref_text: str, cer_id: float, cer_text: float, loss_pc: float):
    """
    Trả về (reject: bool, soft_gate: bool) — bản lenient cho câu ngắn:
      - 1 âm tiết: không hard-reject trừ khi lệch hoàn toàn.
      - 2 âm tiết: nới rộng.
      - >=3 âm tiết: giữ lenient++ như trước.
    """
    L = sum(1 for c in ref_text if 0xAC00 <= ord(c) <= 0xD7A3)
    cer = min(float(cer_text), float(cer_id))

    # === 1 âm tiết: cực kỳ khoan ===
    # Chỉ hard-reject nếu CER ~1.0 và loss rất cao (gần như đọc sai hẳn).
    if L <= 1:
        if cer <= 0.92 or loss_pc <= 22.0:
            return (False, False)   # pass hẳn
        if cer <= 0.98 and loss_pc <= 35.0:
            return (False, True)    # soft gate
        return (True, False)        # hard reject (trường hợp quá lệch)

    # === 2 âm tiết: khá khoan ===
    if L == 2:
        if cer <= 0.75 and loss_pc <= 13.0:
            return (False, False)
        if cer <= 0.92 and loss_pc <= 18.0:
            return (False, True)
        return (True, False)

    # === 3–5 âm tiết: như lenient++ trước đây ===
    if L <= 5:
        if cer <= 0.70 and loss_pc <= 12.0:
            return (False, False)
        if cer <= 0.85 and loss_pc <= 14.0:
            return (False, True)
        return (True, False)

    # === ≥6 âm tiết ===
    if cer <= 0.62 and loss_pc <= 10.0:
        return (False, False)
    if cer <= 0.80 and loss_pc <= 14.0:
        return (False, True)
    return (True, False)

import unicodedata as _ud

def _ctc_to_syllable_confs(ctc_confs: List[float], text: str) -> List[float]:
    """Gộp confidence theo âm tiết Hangeul (Jamo NFD ~ L+V(+T))."""
    # Lấy chỉ các âm tiết Hangul trong text
    syls = [c for c in text if 0xAC00 <= ord(c) <= 0xD7A3]
    out = []
    idx = 0
    for ch in syls:
        nfd = _ud.normalize("NFD", ch)
        # đếm Jamo Hangul (U+1100..11FF)
        j_cnt = sum(1 for x in nfd if 0x1100 <= ord(x) <= 0x11FF)
        if j_cnt <= 0: 
            out.append(0.6);  # fallback
            continue
        seg = ctc_confs[idx: idx + j_cnt] if idx + j_cnt <= len(ctc_confs) else ctc_confs[idx:]
        idx += j_cnt
        if not seg:
            out.append(0.6)
        else:
            out.append(float(np.mean(seg)))
    # pad nếu thiếu
    while len(out) < len(syls): out.append(0.6)
    return out

def score_pronunciation_forced(
    wav_path: Optional[str],
    vectors_user: List[Dict[str, Any]],
    reference_text: str,
    *,
    ref_chunks: List[Tuple[str, List[str]]],
    ref_textgrid_path: Optional[str] = None,
    phone_tier: str = "phones",
    advice_threshold: float = 0.8,
    missing_threshold: float = 0.40,
    min_score_floor: float = 0.0,
    char_confidences: Optional[List[float]] = None,
    # CTC gate params (handled in ctc_gate.py)
    use_ctc_gate: bool = True,
    ctc_loss_char_threshold: float = 2.0,
    ctc_cer_threshold: float = 0.45,
    ctc_mix_mode: str = "min",   # "min" | "geom"
    # NEW: waveform trực tiếp
    waveform: Optional[torch.Tensor] = None,
    waveform_sr: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Forced scoring không cần TG user. Có thể dùng TextGrid ref làm priors.
    Cho phép truyền wav_path HOẶC waveform/sr (ưu tiên waveform).
    """
    # Tổng thời gian
    total_sec = 0.0
    for d in vectors_user:
        total_sec = max(total_sec, float(d.get("end") or 0.0))
    total_sec = max(total_sec, 1e-3)

    ref_chars = [c for c in reference_text] or [" "]

    # ---- CTC gate & per-char confidences (delegated to ctc_gate.py) ----
    ctc_soft_gate_flag = False
    ctc_gate_vals = None
    if use_ctc_gate:
        if waveform is not None and waveform_sr is not None:
            g = ctc_gate_global_from_waveform(
                waveform.squeeze(0) if waveform.ndim == 2 else waveform,
                waveform_sr, reference_text
            )
            ctc_confs = ctc_char_confidences_from_waveform(
                waveform.squeeze(0) if waveform.ndim == 2 else waveform,
                waveform_sr, reference_text
            )
        elif wav_path:
            g = ctc_gate_global(wav_path, reference_text)
            ctc_confs = ctc_char_confidences(wav_path, reference_text)
        else:
            return {"avg_score": 0.0, "details": [], "details_collapsed": [],
                    "params": {"reason": "no_audio_input"},
                    "message": "Thiếu audio đầu vào (wav_path hoặc waveform)."}

        loss_pc = g.get("loss_per_char", 99.0)
        cer_id   = g.get("cer", 1.0)
        cer_text = g.get("cer_text", cer_id)

        # --- Ngưỡng động theo độ dài (chỉ Hangeul)
        reject, soft_gate = _ctc_lenient_gate(reference_text, cer_id, cer_text, loss_pc)

        if reject:
            return {
                "avg_score": 0.0, "details": [], "details_collapsed": [],
                "params": {"reason": "ctc_offtext", "ctc_loss_per_char": loss_pc,
                        "ctc_cer_id": cer_id, "ctc_cer_text": cer_text},
                "message": "Âm thanh không khớp câu tham chiếu (nghi ngờ nói bừa hoặc đọc sai nội dung).",
            }
        if soft_gate:
            char_confidences = (np.array(char_confidences or [], dtype=np.float32) * 0.82).tolist()
            ctc_soft_gate_flag = True
            ctc_gate_vals = {"loss_pc": float(loss_pc), "cer_id": float(cer_id), "cer_text": float(cer_text)}

    # === Pha trộn confidence per-syllable (CTC vs Energy) ===
    try:
        if 'ctc_confs' in locals() and isinstance(ctc_confs, list) and len(ctc_confs) > 0:
            ctc_syl = _ctc_to_syllable_confs(ctc_confs, reference_text)
            # Energy conf hiện đang theo "char" (âm tiết); map chuẩn rồi trộn
            if char_confidences is not None and len(char_confidences) >= len(ctc_syl):
                e = np.asarray(char_confidences[:len(ctc_syl)], dtype=np.float32)
                c = np.asarray(ctc_syl, dtype=np.float32)
                if ctc_mix_mode == "geom":
                    mix = np.sqrt(np.clip(e,1e-6,1.0) * np.clip(c,1e-6,1.0))
                elif ctc_mix_mode == "max":
                    mix = np.maximum(e, c)
                elif ctc_mix_mode == "avg":
                    mix = 0.5 * (e + c)
                else:  # "min" (cũ)
                    mix = np.minimum(e, c)
                # Nới thêm một chút (compressive)
                mix = np.clip(mix, 0.0, 1.0) ** 0.9
                char_confidences = mix.tolist()
    except Exception:
        pass

    # Fallback nếu thiếu
    if char_confidences is None or len(char_confidences) < len(ref_chars):
        fill = [0.8] * len(ref_chars)
        if char_confidences:
            for i in range(min(len(fill), len(char_confidences))):
                fill[i] = float(np.clip(char_confidences[i], 0.0, 1.0))
        char_confidences = fill

    # Chia đều thời gian → char spans
    per = total_sec / len(ref_chars)
    char_spans: List[Tuple[str, float, float, float]] = []
    t = 0.0
    for i, ch in enumerate(ref_chars):
        st = t; en = min(total_sec, t + per); t = en
        p = float(np.clip(char_confidences[i], 0.0, 1.0))
        char_spans.append((ch, st, en, p))

    # chỉ lấy Hangul syllables để map vào ref_chunks
    def _is_hangul_syll(ch: str) -> bool:
        o = ord(ch); return 0xAC00 <= o <= 0xD7A3

    syl_spans_all = [x for x in char_spans if _is_hangul_syll(x[0])]
    N_syl = len(syl_spans_all)
    N_chunk = len(ref_chunks)

    warnings: List[str] = []

    if N_syl == 0 or N_chunk == 0:
        warnings.append(f"empty_spans: syllables={N_syl}, chunks={N_chunk}")

    # Nếu thiếu chunks → pad placeholder để vẫn chấm đủ phần còn lại
    if N_chunk < N_syl:
        deficit = N_syl - N_chunk
        warnings.append(f"pad_chunks: add {deficit} placeholder chunks")
        ref_chunks = ref_chunks + [(f"UNK_{i}", []) for i in range(deficit)]

    # Nếu thừa chunks → cắt nhưng thông báo rõ (hoặc bạn có thể gộp vào cuối tuỳ UX)
    elif N_chunk > N_syl:
        overflow = N_chunk - N_syl
        warnings.append(f"overflow_chunks: drop last {overflow} chunk(s)")
        ref_chunks = ref_chunks[:N_syl]

    # Sau chuẩn hoá, đảm bảo 1–1
    syl_spans = syl_spans_all[:len(ref_chunks)]

    # Priors từ TextGrid ref nếu có
    priors = None
    if ref_textgrid_path and Path(ref_textgrid_path).exists():
        try:
            priors = _build_duration_priors_from_ref_tg(ref_textgrid_path, ref_chunks, phone_tier=phone_tier)
        except Exception as e:
            _log.warning("Build priors from TextGrid failed, fallback equal: %s", e)
            priors = None

    details: List[Dict[str, Any]] = []
    by_chunk: List[Dict[str, Any]] = []
    prev_ph_global = None
    first_syllable_boost = 0.14

    for idx, ((label, phs), (_ch, st, en, p_syl)) in enumerate(zip(ref_chunks, syl_spans)):
        phs_canon = [_canon(p) for p in phs]
        weights = priors[idx] if (priors and idx < len(priors) and len(priors[idx]) == len(phs_canon)) \
                 else _default_phone_weights(phs_canon)
        dur = max(1e-6, en - st)
        bounds = [st]; acc = st
        for w in weights:
            acc += w * dur; bounds.append(acc)

        ph_scores, ph_labels, ph_adv = [], [], []
        for i, ph in enumerate(phs_canon):
            p_st, p_en = bounds[i], bounds[i+1]
            next_ph = phs_canon[i+1] if (i+1) < len(phs_canon) else None
            pos = _estimate_position(ph, prev_ph_global, next_ph)
            w_pos = 1.0 if pos == "nucleus" else (0.995  if pos == "onset" else  0.985)
            raw = float(np.clip(p_syl, 0.0, 1.0)) * w_pos
            if idx == 0: raw = min(1.0, raw + first_syllable_boost)

            missing = p_syl < (missing_threshold * 0.8)
            score = 0.0 if missing else float(max(min_score_floor, min(1.0, raw)))

            adv = advices_for_phoneme(curr=ph, prev=prev_ph_global, next_=next_ph,
                                      position=pos, low_score=(score < advice_threshold), env_hint=None)
            if missing:
                adv = ["Âm này có khả năng bị thiếu (xác suất thấp)."] + adv

            details.append({
                "phoneme": ph, "phoneme_surface": ph, "start": float(p_st), "end": float(p_en),
                "score": score, "color": ("green" if score >= 0.8 else ("yellow" if score >= 0.6 else "red")),
                "advice": adv, "position": pos, "label": label,
            })
            ph_scores.append(score); ph_labels.append(ph); ph_adv.append(adv)
            prev_ph_global = ph

        by_chunk.append({
            "label": label, "phonemes": ph_labels, "scores": ph_scores, "advice": ph_adv,
            "avg_score": (sum(ph_scores)/len(ph_scores) if ph_scores else 0.0),
        })

    avg = float(np.mean([d["score"] for d in details])) if details else 0.0
    return {
        "avg_score": avg,
        "details": details,
        "details_collapsed": by_chunk,
        "params": {
            "advice_threshold": advice_threshold,
            "missing_threshold": missing_threshold,
            "min_score_floor": min_score_floor,
            "ctc_soft_gate": ctc_soft_gate_flag,
            "ctc_gate_values": ctc_gate_vals,
            "warnings": warnings,   # NEW: báo mismatch/pad/drop
        }
    }

# ============ (Tuỳ chọn) Cache sẵn vector & phones cho REF ============
@lru_cache(maxsize=REF_CACHE_MAX)
def precompute_ref_vectors(ref_wav_path: str, ref_tg_path: str):
    feats_r, sr_r, dur_r = extract_features(ref_wav_path)
    phones_r = load_alignment(ref_tg_path)
    vecs_r = extract_phoneme_vectors(feats_r, sr_r, phones_r, dur_r)
    return vecs_r, phones_r

# ============ 9) CLI quick test ============
if __name__ == "__main__":
    init_w2v2()
    dummy = torch.zeros(1, 16000)  # 1s im lặng
    feats, sr, sec = extract_features_from_waveform(dummy, 16000)
    dummy_vecs = [{"phoneme":"_","vector":feats.mean(dim=0),"start":0.0,"end":sec}]
    ref_chunks = [("앉", ["a","n"]), ("아", ["t͡ɕ͈","a"]), ("요", ["j","o"])]
    out = score_pronunciation_forced(
        wav_path=None, waveform=dummy, waveform_sr=16000,
        vectors_user=dummy_vecs, reference_text="앉아요",
        ref_chunks=ref_chunks, ref_textgrid_path=None,
        char_confidences=[0.0, 0.0, 0.0], use_ctc_gate=False  # no CTC in quick test
    )
    import json; print(json.dumps(out, ensure_ascii=False, indent=2))
