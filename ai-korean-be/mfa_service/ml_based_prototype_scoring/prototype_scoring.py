# ml_based_prototype_scoring/prototype_scoring.py
import os
import logging
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple

import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from textgrid import TextGrid

from utils.utils_advice import advices_for_phoneme, VOWELS, STOP_UNRELEASED, PRON_RULES

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

# normalize unreleased stops for RULE CHECK ONLY (không đổi nhãn hiển thị)
UNREL_MAP = {"k̚": "k", "t̚": "t", "p̚": "p"}

# ---- IPA alias / canonicalization ----
_IPA_ALIASES_FROM_RULES = {}
try:
    if isinstance(PRON_RULES, dict):
        _IPA_ALIASES_FROM_RULES = PRON_RULES.get("ipa_aliases", {}) or {}
except Exception:
    _IPA_ALIASES_FROM_RULES = {}

_IPA_ALIASES_DEFAULT = {
    "tɕ": "t͡ɕ", "tɕ͈": "t͡ɕ͈", "tɕʰ": "t͡ɕʰ",
    "e̞": "e", "o̞": "o", "ɐ": "a", "ɫ": "l",
    "ɾ̚": "ɾ",
}

_IPA_ALIASES = {**_IPA_ALIASES_DEFAULT, **_IPA_ALIASES_FROM_RULES}
def _canon(ph: str) -> str:
    return _IPA_ALIASES.get(ph, ph)

def _canon_seq(seq: List[str]) -> List[str]:
    return [_canon(x) for x in seq]

# ============ 2) Feature extraction ============
@torch.inference_mode()
def extract_features(wav_path: str) -> Tuple[torch.Tensor, int, float]:
    init_w2v2()  # đảm bảo đã load & warmup
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

# ============ 3) Load TextGrid ============
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

_DROP_LABELS = {"sp", "sil", "spn", "pau", "ns", "breath", "noise"}
def load_alignment(textgrid_path: str) -> List[Dict[str, Any]]:
    tg = TextGrid()
    tg.read(textgrid_path)
    tier = _pick_phone_tier(tg)
    if tier is None:
        return []
    phones = []
    for itv in tier:
        mark = (itv.mark or "").strip()
        if not mark:
            continue
        if mark.lower() in _DROP_LABELS:
            continue
        phones.append({
            "phoneme": mark,
            "start": float(itv.minTime),
            "end": float(itv.maxTime),
        })
    return phones

# ============ 4) Map time → phoneme vectors ============
@torch.inference_mode()
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
            "start": p.get("start"),
            "end": p.get("end"),
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
    """
    Gán 1 nhãn chunk (từ ref_chunks) cho MỖI phoneme của vectors_ref, theo thứ tự.
    - Nếu ref_chunks is None: trả [""]*len(vectors_ref).
    - Nếu tổng số phone kỳ vọng từ ref_chunks == len(vectors_ref): phân phối 1-1.
    - Nếu lệch: phân phối tỉ lệ, đảm bảo không văng index; chunk cuối ăn phần dư.
    """
    N = len(vectors_ref)
    if not ref_chunks or N == 0:
        return [""] * N

    # Kỳ vọng số phones mỗi chunk theo G2P (đã canonicalize)
    expect_counts = []
    for label, phs in ref_chunks:
        c = sum(1 for p in phs if p and p != "Ø")
        expect_counts.append(max(1, c))  # tối thiểu 1

    tot_expect = sum(expect_counts)
    if tot_expect == N:
        out = []
        idx = 0
        for (label, _), cnt in zip(ref_chunks, expect_counts):
            for _ in range(cnt):
                if idx < N:
                    out.append(label); idx += 1
        # nếu vì lý do nào đó chưa đủ, đệm chunk cuối
        while len(out) < N:
            out.append(ref_chunks[-1][0])
        return out

    # lệch -> phân phối tỉ lệ
    out = []
    idx = 0
    for (label, _), cnt in zip(ref_chunks, expect_counts):
        alloc = int(round((cnt / max(1, tot_expect)) * N))
        # đảm bảo còn dư cho các chunk sau
        rem_chunks = len(ref_chunks) - (len(out) - 0) - 1
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

# ============ 8) Scoring + Advice ============
def score_pronunciation(
    vectors_user: List[Dict[str, Any]],
    vectors_ref: List[Dict[str, Any]],
    phones_user: List[Dict[str, Any]],
    *,
    ref_chunks: Optional[List[Tuple[str, List[str]]]] = None,
    advice_threshold: float = 0.8,
) -> Dict[str, Any]:
    """
    Return:
      {
        "avg_score": float,
        "details": [
          {"phoneme": str, "phoneme_surface": str, "score": float,
           "start": float|None, "end": float|None,
           "color": "green|yellow|red", "advice": [str,...],
           "position": "onset|nucleus|coda|None", "label": <chunk label or "">}
        ],
        "by_chunk": [ {label, phonemes, scores, avg_score, advice} ]
      }
    """
    # 8.0 Lấy nhãn (canonical) từ vectors
    ref_labels_orig = [d["phoneme"] for d in vectors_ref]
    hyp_labels_orig = [d["phoneme"] for d in vectors_user]

    ref_labels = _canon_seq(ref_labels_orig)
    hyp_labels = _canon_seq(hyp_labels_orig)

    # 8.1 NW align theo NHÃN
    pairs = _nw_align_labels(ref_labels, hyp_labels)

    # Map ref_index -> hyp_index (nếu có)
    ref2hyp: Dict[int, Optional[int]] = {}
    for (ri, hj) in pairs:
        if ri is not None:
            ref2hyp[ri] = hj

    # 8.2 Build details (1-1 với REF phones) — không bỏ sót phoneme nào
    details: List[Dict[str, Any]] = []
    for i, (r_lab, r_raw) in enumerate(zip(ref_labels, ref_labels_orig)):
        hj = ref2hyp.get(i, None)
        if hj is not None and hj < len(vectors_user):
            sim = cosine_score_torch(vectors_user[hj]["vector"], vectors_ref[i]["vector"])
            matched_label = hyp_labels[hj]
        else:
            sim = 0.0
            matched_label = None

        color = "green" if sim >= 0.8 else ("yellow" if sim >= 0.6 else "red")

        prev_ref = ref_labels[i - 1] if i > 0 else None
        next_ref = ref_labels[i + 1] if i + 1 < len(ref_labels) else None
        position = _estimate_position(r_lab, prev_ref, next_ref)

        # gợi ý
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
            curr=r_lab,
            prev=prev_ref,
            next_=next_ref,
            position=position,
            low_score=(sim < advice_threshold),
            env_hint=env_hint,
        )

        # hiển thị coda unreleased nếu nhãn gốc có (để người học thấy k̚/t̚/p̚)
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

    # 8.3 (tuỳ chọn) gán CHUNK label theo ref_chunks
    by_chunk: List[Dict[str, Any]] = []
    if ref_chunks:
        chunk_labels_seq = _build_chunk_labels_for_ref(vectors_ref, ref_chunks)
        # gắn label vào details
        for d, lab in zip(details, chunk_labels_seq):
            d["label"] = lab

        # gom theo label liên tiếp
        cur = None
        cur_ph, cur_sc, cur_adv = [], [], []
        for d in details:
            lab = d.get("label", "")
            if cur is None:
                cur = lab
            if lab != cur:
                # flush
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
        # nếu không có chunk, set rỗng để FE dễ xử lý
        for d in details:
            d["label"] = ""

    result = {
        "avg_score": avg,
        "details": details,
        "details_collapsed": by_chunk,
        "pairs": pairs,  # để debug
    }
    return result

# ============ 9) (Tuỳ chọn) Cache sẵn vector & phones cho REF ============
@lru_cache(maxsize=REF_CACHE_MAX)
def precompute_ref_vectors(ref_wav_path: str, ref_tg_path: str):
    feats_r, sr_r, dur_r = extract_features(ref_wav_path)
    phones_r = load_alignment(ref_tg_path)
    vecs_r = extract_phoneme_vectors(feats_r, sr_r, phones_r, dur_r)
    return vecs_r, phones_r

# ============ 10) CLI quick test ============
if __name__ == "__main__":
    init_w2v2()
    # Put your quick local test here if needed.
    pass
