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
    """Load & warmup Wav2Vec2 (idempotent). Gọi ở FastAPI startup để giảm latency."""
    global _processor, _model, _warmed_up
    if _processor is None:
        _processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    if _model is None:
        _model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    if not _warmed_up:
        try:
            with torch.inference_mode():
                dummy = torch.zeros(1, TARGET_SR, dtype=torch.float32, device=DEVICE)
                _ = _model(dummy)  # warm weights & kernels
            _warmed_up = True
            _log.info("W2V2 warmup done for %s on %s (threads=%d)", MODEL_NAME, DEVICE, NUM_THREADS)
        except Exception as e:
            _log.warning("Warmup failed (safe to ignore on some setups): %s", e)
            _warmed_up = True  # tránh warmup lại nhiều lần

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

# IPA alias/canonicalization (để ALIGN & priors)
ALIAS_CANON: Dict[str, str] = {
    "tɕ": "t͡ɕ", "tɕ͈": "t͡ɕ͈", "tɕʰ": "t͡ɕʰ",
    "e̞": "e", "o̞": "o", "ɐ": "a", "ɫ": "l",
    "ɾ̚": "ɾ",
}

def _canon(ph: str) -> str:
    return ALIAS_CANON.get(ph, ph)

def _canon_seq(seq: List[str]) -> List[str]:
    return [_canon(x) for x in seq]

# ============ 2) Feature extraction ============
@torch.inference_mode()
def extract_features(wav_path: str) -> Tuple[torch.Tensor, int, float]:
    init_w2v2()  # đảm bảo đã load
    waveform, sr = torchaudio.load(wav_path)  # [C, N]
    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        resampler = _get_resampler(sr, TARGET_SR)
        waveform = resampler(waveform) if resampler is not None else waveform
        sr = TARGET_SR
    mono = waveform.squeeze(0)
    inputs = _processor(mono, sampling_rate=sr, return_tensors="pt").input_values.to(DEVICE)  # type: ignore
    outputs = _model(inputs)  # type: ignore
    feats = outputs.last_hidden_state.squeeze(0).detach().cpu()  # [T, D]
    total_audio_sec = float(mono.shape[0]) / float(sr)
    return feats, sr, total_audio_sec

# ========= 2b) Ước lượng char_confidences từ năng lượng =========
def estimate_char_confidences_from_energy(
    wav_path: str,
    reference_text: str,
    *,
    sr_target: int = TARGET_SR,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
    vad_lo_q: float = 0.15,      # ngưỡng VAD (lọc im lặng)
    # SOFT-VOICED thay vì nhị phân: dùng 2 phân vị để chuẩn hoá mềm
    soft_q_lo: float = 0.2,
    soft_q_hi: float = 0.8,
    # Đệm đầu/cuối vùng thoại (giảm "tụt" âm đầu/cuối)
    lead_pad_ms: float = 200.0,
    tail_pad_ms: float = 100.0,
    edge_comp: float = 1.0,      # bỏ phạt mép
    smooth_win: int = 7,         # mượt hơn

) -> List[float]:
    import numpy as np
    import torchaudio

    # Load → mono → resample
    wav, sr = torchaudio.load(wav_path)
    if wav.ndim == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sr_target:
        resampler = _get_resampler(sr, sr_target)
        wav = resampler(wav) if resampler is not None else wav
        sr = sr_target
    x = wav.squeeze(0).numpy().astype(np.float32)
    Nchar = max(1, len(reference_text))
    if x.size == 0:
        return [0.0] * Nchar

    # Framing + RMS → log-RMS (dB-like)
    frame = max(1, int(round(frame_ms * sr / 1000.0)))
    hop = max(1, int(round(hop_ms * sr / 1000.0)))
    frames = [x[i:i+frame] for i in range(0, max(1, len(x) - frame + 1), hop)]
    if not frames: frames = [x]
    mat = np.stack(frames, axis=0)                # [F, L]
    rms = np.sqrt(np.mean(mat * mat, axis=1))     # [F]
    log_rms = 20.0 * np.log10(np.maximum(rms, 1e-8))

    # VAD thô bằng phân vị log-RMS (ổn định hơn tuyệt đối)
    thr_vad = np.quantile(log_rms, vad_lo_q)
    idx = np.where(log_rms > thr_vad)[0]
    if idx.size == 0:
        return [0.0] * Nchar
    # MỞ RỘNG vùng thoại về trái/phải để bảo vệ âm đầu/cuối
    i0, i1 = int(idx[0]), int(idx[-1]) + 1
    hop = max(1, int(round(hop_ms * sr / 1000.0)))
    lead = max(0, int(round(lead_pad_ms * sr / 1000.0 / hop)))
    tail = max(0, int(round(tail_pad_ms * sr / 1000.0 / hop)))
    s = max(0, i0 - lead)
    e = min(len(log_rms), i1 + tail)
    log_rms = log_rms[s:e]
    if log_rms.size < Nchar:
        return [0.0] * Nchar

    # SOFT-VOICED: scale 0..1 giữa 2 phân vị
    lo = np.quantile(log_rms, soft_q_lo)
    hi = np.quantile(log_rms, soft_q_hi)
    denom = max(1e-6, (hi - lo))
    voiced = np.clip((log_rms - lo) / denom, 0.0, 1.0).astype(np.float32)

    # Chia đều thời gian vùng thoại theo số ký tự
    F = len(log_rms)
    per = F / float(Nchar)
    bounds = [0]
    for k in range(1, Nchar):
        bounds.append(int(round(k * per)))
    bounds.append(F)

    # Confidence = trung bình voiced trong span
    confs = []
    for k in range(Nchar):
        a, b = bounds[k], bounds[k+1]
        if b <= a:
            confs.append(0.0)
        else:
            frac = float(voiced[a:b].mean())
            confs.append(frac)

    # Bù mép + làm mượt
    if Nchar >= 1:
        confs[0] *= edge_comp
        confs[-1] *= edge_comp
    if smooth_win > 1 and Nchar > 2:
        k = np.ones(smooth_win, dtype=np.float32) / smooth_win
        confs = list(np.convolve(np.array(confs, dtype=np.float32), k, mode="same"))

    # Gamma-compress + dynamic blend nếu mean thấp + boost nhẹ 2 ký tự đầu
    confs = np.asarray(confs, dtype=np.float32)
    confs = np.clip(confs, 0.0, 1.0) ** 0.6

    m = float(confs.mean()) if confs.size else 0.0
    beta = float(np.clip((0.68 - m) / 0.20, 0.0, 0.6))  # chỉ blend khi mean thấp
    if beta > 0.0:
        confs = (1.0 - beta) * confs + beta * 0.7

    if confs.size >= 1: confs[0] = min(1.0, confs[0] + 0.05)
    if confs.size >= 2: confs[1] = min(1.0, confs[1] + 0.03)

    confs = np.clip(confs, 0.0, 1.0).astype(np.float32).tolist()
    return confs

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
    tg = TextGrid()
    tg.read(textgrid_path)
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
    ref_labels: List[str],
    hyp_labels: List[str],
    match_score: float = 1.0,
    mismatch_penalty: float = -0.5,
    gap_penalty: float = -0.6,
) -> List[Tuple[Optional[int], Optional[int]]]:
    n, m = len(ref_labels), len(hyp_labels)
    dp = [[0.0]*(m+1) for _ in range(n+1)]
    bt = [[0]*(m+1) for _ in range(n+1)]  # 0 diag, 1 up (del), 2 left (ins)

    for i in range(1, n+1):
        dp[i][0] = dp[i-1][0] + gap_penalty; bt[i][0] = 1
    for j in range(1, m+1):
        dp[0][j] = dp[0][j-1] + gap_penalty; bt[0][j] = 2

    for i in range(1, n+1):
        for j in range(1, m+1):
            d = dp[i-1][j-1] + (match_score if ref_labels[i-1] == hyp_labels[j-1] else mismatch_penalty)
            u = dp[i-1][j]   + gap_penalty
            l = dp[i][j-1]   + gap_penalty
            if d >= u and d >= l:
                dp[i][j] = d; bt[i][j] = 0
            elif u >= l:
                dp[i][j] = u; bt[i][j] = 1
            else:
                dp[i][j] = l; bt[i][j] = 2

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

# ============ 7) Chunk label mapping (ref_chunks -> ref vectors) ============
def _build_chunk_labels_for_ref(
    vectors_ref: List[Dict[str, Any]],
    ref_chunks: Optional[List[Tuple[str, List[str]]]],
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
                if idx < N:
                    out.append(label); idx += 1
        while len(out) < N:
            out.append(ref_chunks[-1][0])
        return out
    out, idx = [], 0
    for (label, _), cnt in zip(ref_chunks, expect_counts):
        alloc = int(round((cnt / max(1, tot_expect)) * N))
        rem_chunks = len(ref_chunks) - len(out) - 1
        rem_slots = N - idx
        if rem_chunks <= 0:
            alloc = rem_slots
        else:
            alloc = min(alloc, rem_slots - rem_chunks)
        alloc = max(1, alloc)
        for _ in range(alloc):
            if idx < N:
                out.append(label); idx += 1
    while len(out) < N:
        out.append(ref_chunks[-1][0])
    return out

# ============ 8A) Scoring + Advice (MFA-W2V; cần TextGrid user) ============
def score_pronunciation(
    vectors_user: List[Dict[str, Any]],
    vectors_ref: List[Dict[str, Any]],
    phones_user: List[Dict[str, Any]],
    *,
    ref_chunks: Optional[List[Tuple[str, List[str]]]] = None,
    advice_threshold: float = 0.8,
) -> Dict[str, Any]:
    ref_labels_orig = [d["phoneme"] for d in vectors_ref]
    hyp_labels_orig = [d["phoneme"] for d in vectors_user]
    ref_labels = _canon_seq(ref_labels_orig)
    hyp_labels = _canon_seq(hyp_labels_orig)
    pairs = _nw_align_labels(ref_labels, hyp_labels)
    ref2hyp: Dict[int, Optional[int]] = {}
    for (ri, hj) in pairs:
        if ri is not None:
            ref2hyp[ri] = hj
    details: List[Dict[str, Any]] = []
    for i, (r_lab, r_raw) in enumerate(zip(ref_labels, ref_labels_orig)):
        hj = ref2hyp.get(i, None)
        missing = False
        if hj is not None and hj < len(vectors_user):
            sim = cosine_score_torch(vectors_user[hj]["vector"], vectors_ref[i]["vector"])
            matched_label = hyp_labels[hj]
        else:
            sim = 0.0
            matched_label = None
            missing = True
        color = "green" if sim >= 0.8 else ("yellow" if sim >= 0.6 else "red")
        prev_ref = ref_labels[i - 1] if i > 0 else None
        next_ref = ref_labels[i + 1] if i + 1 < len(ref_labels) else None
        position = _estimate_position(r_lab, prev_ref, next_ref)
        env_hint = None
        if matched_label is not None and matched_label != r_lab:
            if ("͈" in r_lab) != ("͈" in matched_label):
                env_hint = "tense_diff"
            elif (r_lab.endswith("ʰ")) != (matched_label.endswith("ʰ")):
                env_hint = "asp_diff"
            elif (r_lab in {"k̚","t̚","p̚"}) != (matched_label in {"k̚","t̚","p̚"}):
                env_hint = "release_diff"
            elif (r_lab in {"j","w"}) != (matched_label in {"j","w"}):
                env_hint = "glide_diff"
        tips = advices_for_phoneme(
            curr=r_lab, prev=prev_ref, next_=next_ref,
            position=position, low_score=(sim < advice_threshold),
            env_hint=env_hint,
        )
        if missing:
            tips = ["Âm này bị thiếu (bỏ/nuốt âm hoặc không khớp)."] + tips
        display_label = r_lab
        if position == "coda" and r_raw in STOP_UNRELEASED:
            display_label = r_raw
        details.append({
            "phoneme": display_label,
            "phoneme_surface": r_lab,
            "score": round(float(sim), 4),
            "start": float(vectors_ref[i].get("start")) if vectors_ref[i].get("start") is not None else None,
            "end": float(vectors_ref[i].get("end")) if vectors_ref[i].get("end") is not None else None,
            "color": color,
            "advice": tips,
            "position": position,
        })
    avg = float(np.mean([d["score"] for d in details])) if details else 0.0
    by_chunk: List[Dict[str, Any]] = []
    if ref_chunks:
        chunk_labels_seq = _build_chunk_labels_for_ref(vectors_ref, ref_chunks)
        for d, lab in zip(details, chunk_labels_seq):
            d["label"] = lab
        cur = None
        cur_ph, cur_sc, cur_adv = [], [], []
        for d in details:
            lab = d.get("label", "")
            if cur is None:
                cur = lab
            if lab != cur:
                if cur_ph:
                    avg_c = float(np.mean(cur_sc)) if cur_sc else 0.0
                    by_chunk.append({
                        "label": cur,
                        "phonemes": cur_ph,
                        "scores": cur_sc,
                        "avg_score": avg_c,
                        "advice": list(dict.fromkeys(cur_adv))[:3],
                    })
                cur = lab
                cur_ph, cur_sc, cur_adv = [], [], []
            cur_ph.append(d["phoneme"]) 
            cur_sc.append(d["score"]) 
            cur_adv.extend(d.get("advice") or [])
        if cur_ph:
            avg_c = float(np.mean(cur_sc)) if cur_sc else 0.0
            by_chunk.append({
                "label": cur or "",
                "phonemes": cur_ph,
                "scores": cur_sc,
                "avg_score": avg_c,
                "advice": list(dict.fromkeys(cur_adv))[:3],
            })
    else:
        for d in details:
            d["label"] = ""
    return {
        "avg_score": avg,
        "details": details,
        "details_collapsed": by_chunk,
        "pairs": pairs,
    }

# ============ 8B) Scoring (W2v2Forced; không cần TG user; optional TG ref) ============
def _build_duration_priors_from_ref_tg(
    ref_textgrid_path: str,
    ref_chunks: List[Tuple[str, List[str]]],
    *,
    phone_tier: str = "phones",
    drop_labels: Optional[set] = None,
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
        if lab in drop_labels:
            continue
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
        if lens <= 0:
            priors.append([])
            continue
        local = dur_per_g2p[s:e]
        total = sum(local)
        if total <= 1e-8:
            priors.append([1.0 / lens] * lens)
        else:
            priors.append([x / total for x in local])
    return priors


def _default_phone_weights(phones: List[str]) -> List[float]:
    if not phones:
        return []
    vals = []
    for ph in phones:
        if _is_vowel(ph):
            vals.append(1.5)
        elif ph in {"j", "w"}:
            vals.append(1.1)
        else:
            vals.append(1.0)
    s = sum(vals)
    return [v / s for v in vals]


def score_pronunciation_forced(
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
) -> Dict[str, Any]:
    """
    Không cần TextGrid của user để chạy.
    Nếu có ref_textgrid_path: dùng làm duration priors cho từng chunk.
    Có thể truyền char_confidences (0..1) theo từng ký tự của reference_text để gate thiếu âm.
    """
    # Tổng thời gian
    total_sec = 0.0
    for d in vectors_user:
        total_sec = max(total_sec, float(d.get("end") or 0.0))
    total_sec = max(total_sec, 1e-3)

    # Char confidences: nếu không có, fallback = 0.7 (giữ backward-compat)
    ref_chars = [c for c in reference_text] or [" "]
    if char_confidences is None or len(char_confidences) < len(ref_chars):
        fill = [0.7] * len(ref_chars)
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
        o = ord(ch)
        return 0xAC00 <= o <= 0xD7A3

    syl_spans = [x for x in char_spans if _is_hangul_syll(x[0])]
    K = min(len(syl_spans), len(ref_chunks))
    syl_spans = syl_spans[:K]
    ref_chunks = ref_chunks[:K]

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
    first_syllable_boost = 0.06  # cộng nhẹ vào raw score âm tiết đầu

    for idx, ((label, phs), (_ch, st, en, p_syl)) in enumerate(zip(ref_chunks, syl_spans)):
        phs_canon = [_canon(p) for p in phs]
        if priors and idx < len(priors) and len(priors[idx]) == len(phs_canon):
            weights = priors[idx]
        else:
            weights = _default_phone_weights(phs_canon)

        dur = max(1e-6, en - st)
        bounds = [st]; acc = st
        for w in weights:
            acc += w * dur
            bounds.append(acc)

        ph_scores, ph_labels, ph_adv = [], [], []
        for i, ph in enumerate(phs_canon):
            p_st, p_en = bounds[i], bounds[i+1]
            next_ph = phs_canon[i+1] if (i+1) < len(phs_canon) else None
            pos = _estimate_position(ph, prev_ph_global, next_ph)

            # Phạt vị trí nhẹ (ổn định hơn so với 1.0/1.0/1.0)
            w_pos = 1.0 if pos == "nucleus" else (0.97 if pos == "onset" else 0.94)
            raw = float(np.clip(p_syl, 0.0, 1.0)) * w_pos
            if idx == 0:
                raw = min(1.0, raw + first_syllable_boost)  # bù nhẹ âm tiết đầu

            # GATE thiếu âm (dựa trên p_syl thô)
            missing = p_syl < missing_threshold
            score = 0.0 if missing else float(max(min_score_floor, min(1.0, raw)))

            adv = advices_for_phoneme(
                curr=ph, prev=prev_ph_global, next_=next_ph,
                position=pos, low_score=(score < advice_threshold), env_hint=None
            )
            if missing:
                adv = ["Âm này có khả năng bị thiếu (xác suất thấp)."] + adv

            details.append({
                "phoneme": ph,
                "phoneme_surface": ph,
                "start": float(p_st),
                "end": float(p_en),
                "score": score,
                "color": ("green" if score >= 0.8 else ("yellow" if score >= 0.6 else "red")),
                "advice": adv,
                "position": pos,
                "label": label,
            })
            ph_scores.append(score)
            ph_labels.append(ph)
            ph_adv.append(adv)
            prev_ph_global = ph

        by_chunk.append({
            "label": label,
            "phonemes": ph_labels,
            "scores": ph_scores,
            "advice": ph_adv,
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
    dummy_vecs = [{"phoneme":"_","vector":torch.zeros(10),"start":0.0,"end":2.0}]
    ref_chunks = [("앉", ["a","n"]), ("아", ["t͡ɕ͈","a"]), ("요", ["j","o"])]
    # ví dụ: im lặng → char_confidences toàn 0 → score=0
    out = score_pronunciation_forced(
        dummy_vecs, "앉아요",
        ref_chunks=ref_chunks, ref_textgrid_path=None,
        char_confidences=[0.0, 0.0, 0.0]
    )
    import json
    print(json.dumps(out, ensure_ascii=False, indent=2))
