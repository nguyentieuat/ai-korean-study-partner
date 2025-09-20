# ctc_gate.py — CTC-only gate & confidences for Korean (NFD/Compat Jamo)
# - Blank-posterior VAD (no energy)
# - Viterbi-local CTC for per-jamo confidences
# - Multi-signal gating: voiced, avg top1, coverage, repetition, duration, CER/neglogp
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
MODEL_NAME = os.getenv("W2V_MODEL", "Kkonjeong/wav2vec2-base-korean")  # NFD/Compat đều OK, code tự map
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR = 16000
VAD_PAD_MS = float(os.getenv("W2V_VAD_PAD_MS", "120.0"))
BLANK_VAD_THRESH = float(os.getenv("W2V_BLANK_VAD_THRESH", "0.7"))
LOGIT_TEMP = float(os.getenv("W2V_LOGIT_TEMP", "1.0"))

# Gating thresholds (có thể override bằng env)
VOICED_MIN = float(os.getenv("W2V_GATE_VOICED_MIN", "0.20"))        # >= 20% khung non-blank
TOP1_NONBLANK_MIN = float(os.getenv("W2V_GATE_TOP1_MIN", "0.35"))    # >= 0.35 trung bình top-1 (non-blank)
COVERAGE_MIN = float(os.getenv("W2V_GATE_COVERAGE_MIN", "0.40"))     # >= 40% jamo có conf > missing_thr
REP_RUN_MAX = float(os.getenv("W2V_GATE_REPRUN_MAX", "0.85"))        # <= 85% cho chuỗi lặp dài nhất
DUR_PER_JAMO_MS = float(os.getenv("W2V_GATE_DUR_PER_JAMO_MS", "70")) # 70ms/jamo kỳ vọng
DUR_TOL_LOW = float(os.getenv("W2V_GATE_DUR_TOL_LOW", "0.45"))       # voiced >= 45% expected
DUR_TOL_HIGH = float(os.getenv("W2V_GATE_DUR_TOL_HIGH", "2.5"))      # voiced <= 2.5x expected
MISSING_THR = float(os.getenv("W2V_MISSING_THRESHOLD", "0.08"))

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
        _log.info(f"Loading W2V2 model: {MODEL_NAME}")
        _processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        _model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(DEVICE)
        _model.eval()
        _blank_id = int(getattr(_model.config, "pad_token_id", 0) or 0)

def _log_softmax(logits: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
    if temp != 1.0:
        logits = logits / max(1e-6, temp)
    return torch.log_softmax(logits, dim=-1)

def _blank_posterior_vad(logprobs: torch.Tensor, blank_id: int, thresh: float = 0.7, pad_ms: float = 120.0) -> slice:
    with torch.no_grad():
        blank_post = logprobs.exp()[:, blank_id]  # [T]
        non_blank = (blank_post < thresh).nonzero(as_tuple=False).squeeze(-1)
        if non_blank.numel() == 0:
            return slice(0, logprobs.shape[0])
        start = non_blank[0].item()
        end = non_blank[-1].item() + 1
        pad_frames = int(round(pad_ms / 20.0))
        start = max(0, start - pad_frames)
        end = min(logprobs.shape[0], end + pad_frames)
        return slice(start, end)

# ============ Jamo helpers ============
def _to_nfd_jamo(text: str) -> List[str]:
    return list(text)

# NFD <-> Compatibility mapping (để chịu được cả 2 loại vocab)
import unicodedata as _ud
_LEADS_NFD  = "ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄋᄌᄍᄎᄏᄐᄑᄒ"
_LEADS_COMP = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_VOWS_NFD   = "ᅡᅢᅣᅤᅥᅦᅧᅨᅩᅪᅫᅬᅭᅮᅯᅰᅱᅲᅳᅴᅵ"
_VOWS_COMP  = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_TRLS_NFD   = ["ᆨ","ᆩ","ᆪ","ᆫ","ᆬ","ᆭ","ᆮ","ᆯ","ᆰ","ᆱ","ᆲ","ᆳ","ᆴ","ᆵ","ᆶ","ᆷ","ᆸ","ᆹ","ᆺ","ᆻ","ᆼ","ᆽ","ᆾ","ᆿ","ᇀ","ᇁ","ᇂ"]
_TRLS_COMP  = ["ㄱ","ㄲ","ㄳ","ㄴ","ㄵ","ㄶ","ㄷ","ㄹ","ㄺ","ㄻ","ㄼ","ㄽ","ㄾ","ㄿ","ㅀ","ㅁ","ㅂ","ㅄ","ㅅ","ㅆ","ㅇ","ㅈ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"]
_NFD2COMP = {n:c for n,c in zip(_LEADS_NFD, _LEADS_COMP)}
_NFD2COMP.update({n:c for n,c in zip(_VOWS_NFD, _VOWS_COMP)})
_NFD2COMP.update({n:c for n,c in zip(_TRLS_NFD, _TRLS_COMP)})

def _map_char_to_vocab(ch: str, vocab: Dict[str,int]) -> Optional[str]:
    if ch in vocab:
        return ch
    alt = _NFD2COMP.get(ch)
    if alt and (alt in vocab):
        return alt
    return None

def _map_ref_jamo_seq_to_vocab(ref_jamo_seq: List[str], vocab: Dict[str,int]) -> Tuple[List[str], List[int]]:
    tokens_mapped, pos_map = [], []
    for t in ref_jamo_seq:
        if not t or t.isspace():
            pos_map.append(-1); continue
        mt = _map_char_to_vocab(t, vocab)
        if mt is None:
            pos_map.append(-1)
        else:
            pos_map.append(len(tokens_mapped))
            tokens_mapped.append(mt)
    return tokens_mapped, pos_map

def _build_confusable_ids_dynamic(vocab: Dict[str,int]) -> Optional[Dict[int, List[int]]]:
    base = {
        "ᄂ": ["ᄅ"], "ᄅ": ["ᄂ"], "ᄇ": ["ᄑ"], "ᄑ": ["ᄇ"],
        "ᄃ": ["ᄐ","ᄌ","ᄎ"], "ᄐ": ["ᄃ","ᄌ","ᄎ"], "ᄌ": ["ᄃ","ᄐ","ᄎ"], "ᄎ": ["ᄃ","ᄐ","ᄌ"],
        "ㄴ": ["ㄹ"], "ㄹ": ["ㄴ"], "ㅂ": ["ㅍ"], "ㅍ": ["ㅂ"],
        "ㄷ": ["ㅌ","ㅈ","ㅊ"], "ㅌ": ["ㄷ","ㅈ","ㅊ"], "ㅈ": ["ㄷ","ㅌ","ㅊ"], "ㅊ": ["ㄷ","ㅌ","ㅈ"],
    }
    out: Dict[int, List[int]] = {}
    for t, alts in base.items():
        if t in vocab:
            ids = [vocab[a] for a in alts if a in vocab]
            if ids:
                out[vocab[t]] = sorted(set(ids))
    return out or None

# ========= Viterbi-local CTC =========
def _expand_ref_with_blanks(ref_ids: List[int], blank_id: int) -> List[int]:
    out = []
    for tid in ref_ids:
        out.append(blank_id)
        out.append(tid)
    out.append(blank_id)
    return out

def _forced_ctc_viterbi_path(logprobs: torch.Tensor, ext: List[int]) -> torch.Tensor:
    T, M = logprobs.shape[0], len(ext)
    dp = torch.full((T, M), -1e9, device=logprobs.device)
    bp = torch.full((T, M), -1, device=logprobs.device, dtype=torch.long)
    dp[0, 0] = logprobs[0, ext[0]]
    if M > 1:
        dp[0, 1] = logprobs[0, ext[1]]
        bp[0, 1] = 0
    for t in range(1, T):
        stay = dp[t-1]
        prev = torch.cat([torch.tensor([-1e9], device=dp.device), dp[t-1, :-1]], dim=0)
        prev2 = torch.cat([torch.tensor([-1e9, -1e9], device=dp.device), dp[t-1, :-2]], dim=0)
        mask = torch.ones(M, dtype=torch.bool, device=dp.device)
        mask[2:] = torch.tensor([ext[m] != ext[m-2] for m in range(2, M)], device=dp.device)
        prev2 = torch.where(mask, prev2, torch.full_like(prev2, -1e9))
        cand = torch.stack([stay, prev, prev2], 0)
        best, arg = cand.max(0)
        dp[t] = best + logprobs[t, ext]
        bp[t] = arg
    m = M-1 if M > 1 else 0
    if M > 1 and dp[T-1, M-2] > dp[T-1, M-1]:
        m = M-2
    m_path = torch.empty(T, dtype=torch.long, device=dp.device)
    m_path[-1] = m
    for t in range(T-2, -1, -1):
        move = bp[t+1, m]
        if move == 1: m -= 1
        elif move == 2: m -= 2
        m_path[t] = m
    return m_path

def _forced_ctc_jamo_confidences(logprobs: torch.Tensor, ref_ids: List[int], blank_id: int,
                                 confusable_ids: Optional[Dict[int, List[int]]] = None) -> List[float]:
    ext = _expand_ref_with_blanks(ref_ids, blank_id)
    m_path = _forced_ctc_viterbi_path(logprobs, ext)
    confs: List[float] = []
    for i in range(len(ref_ids)):
        m_idx = 2*i + 1
        mask = (m_path == m_idx)
        if not mask.any():
            confs.append(0.0); continue
        seg = logprobs[mask]
        num = seg[:, ext[m_idx]].exp().sum()
        den = (1.0 - seg[:, blank_id].exp()).clamp_min(1e-6).sum()
        ci = float((num / den).item())
        if confusable_ids and ref_ids[i] in confusable_ids:
            alts = confusable_ids[ref_ids[i]]
            if len(alts) > 0:
                alt_mass = seg[:, alts].exp().sum()
                ci = float(np.clip(ci + 0.5 * float((alt_mass / den).item()), 0.0, 1.0))
        confs.append(ci)
    return confs

# ========= Extra gating features =========
def _gate_features(logprobs: torch.Tensor, vocab: Dict[str,int], ref_ids: List[int]) -> Dict[str, float]:
    with torch.no_grad():
        probs = logprobs.exp()                           # [T, V]
        blank = probs[:, _blank_id]                      # [T]
        nonblank = 1.0 - blank                           # [T]
        T = probs.shape[0]
        # voiced ratio
        voiced_ratio = float((nonblank > 0.3).float().mean().item())
        # avg top1 (non-blank frames)
        top1_vals, top1_ids = probs.max(dim=-1)          # [T]
        nb_mask = (top1_ids != _blank_id)
        avg_top1_nonblank = float((top1_vals[nb_mask].mean().item() if nb_mask.any() else 0.0))
        # repetition run longest/T
        if T > 0:
            ids = top1_ids.cpu().numpy().tolist()
            longest = 1; cur = 1
            for i in range(1, T):
                if ids[i] == ids[i-1]:
                    cur += 1
                    if cur > longest: longest = cur
                else:
                    cur = 1
            repetition_run = float(longest / T)
        else:
            repetition_run = 1.0
        # voiced sec (xấp xỉ 20ms/frame)
        voiced_sec = float(nonblank.sum().item() * 0.02)
        return {
            "voiced_ratio": voiced_ratio,
            "avg_top1_nonblank": avg_top1_nonblank,
            "repetition_run": repetition_run,
            "voiced_sec": voiced_sec,
        }

# ============ Public API ============
def ctc_char_confidences_from_waveform(
    waveform: torch.Tensor,
    sr: int,
    ref_text_nfd: str,
    blank_vad_thresh: float = None,
    vad_pad_ms: float = None,
    temp: float = None,
    use_confusables: bool = True,
    ref_jamo_seq: Optional[List[str]] = None,
) -> List[float]:
    _init()
    blank_thr = BLANK_VAD_THRESH if blank_vad_thresh is None else blank_vad_thresh
    pad_ms = VAD_PAD_MS if vad_pad_ms is None else vad_pad_ms
    Ttemp = LOGIT_TEMP if temp is None else temp

    wav = _ensure_sr(waveform, sr)
    with torch.inference_mode():
        inputs = _processor(wav.squeeze(0), sampling_rate=TARGET_SR, return_tensors="pt", padding=False)
        input_values = inputs.input_values.to(DEVICE)
        logits = _model(input_values).logits[0]
        logprobs = _log_softmax(logits, temp=Ttemp)

        vslice = _blank_posterior_vad(logprobs, _blank_id, thresh=blank_thr, pad_ms=pad_ms)
        logprobs = logprobs[vslice]

        vocab = _processor.tokenizer.get_vocab()
        if ref_jamo_seq is None:
            ref_tokens_src = [t for t in _to_nfd_jamo(ref_text_nfd) if t.strip() != ""]
        else:
            ref_tokens_src = [t for t in ref_jamo_seq if t and not t.isspace()]

        mapped_tokens, pos_map = _map_ref_jamo_seq_to_vocab(ref_tokens_src, vocab)
        if len(mapped_tokens) == 0 or logprobs.shape[0] == 0:
            return [0.0] * max(1, len(ref_tokens_src))

        ref_ids = [vocab[tok] for tok in mapped_tokens]
        conf_ids = _build_confusable_ids_dynamic(vocab) if use_confusables else None

        confs_mapped = _forced_ctc_jamo_confidences(logprobs, ref_ids, _blank_id, confusable_ids=conf_ids)

        out: List[float] = []
        for p in pos_map:
            out.append(float(confs_mapped[p]) if p != -1 else 0.0)
        return out

def ctc_char_confidences(wav_path: str, ref_text_nfd: str, **kwargs) -> List[float]:
    wav, sr = torchaudio.load(wav_path)
    return ctc_char_confidences_from_waveform(wav, sr, ref_text_nfd, **kwargs)

def _best_path_decode(logprobs: torch.Tensor, vocab_id2tok: Dict[int, str]) -> str:
    ids = logprobs.argmax(dim=-1).tolist()
    out = []
    prev = None
    for i in ids:
        if i == _blank_id:
            prev = i; continue
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

def ctc_gate_global_from_waveform(
    waveform: torch.Tensor,
    sr: int,
    ref_text_nfd: str,
    cer_threshold: float = 0.65,
    loss_char_threshold: float = 2.0
) -> Dict[str, float]:
    """Multi-signal gate; trả về các chỉ số + cờ gate_pass."""
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
        hyp = _best_path_decode(logprobs, id2tok)

        print("Hyp:", hyp)

        ref_tokens_src = [t for t in _to_nfd_jamo(ref_text_nfd) if t.strip() != ""]
        mapped_tokens, _ = _map_ref_jamo_seq_to_vocab(ref_tokens_src, vocab)
        ref_mapped = "".join(mapped_tokens)

        cer = _cer(hyp, ref_mapped)

        # mean neglogp (trên ref tokens)
        ref_ids = [vocab[ch] for ch in mapped_tokens if ch in vocab]
        if len(ref_ids) == 0 or logprobs.shape[0] == 0:
            mean_neglogp = 10.0
        else:
            tok_post = logprobs.exp().mean(0)
            pick = tok_post[ref_ids].clamp_min(1e-6)
            mean_neglogp = float((-pick.log()).mean().item())

        # extra gate signals
        gf = _gate_features(logprobs, vocab, ref_ids)
        voiced_ratio = gf["voiced_ratio"]
        avg_top1_nonblank = gf["avg_top1_nonblank"]
        repetition_run = gf["repetition_run"]
        voiced_sec = gf["voiced_sec"]

        # coverage ratio từ Viterbi-local confidences
        coverage_ratio = 0.0
        if ref_ids:
            conf_ids = _build_confusable_ids_dynamic(vocab)
            confs_ref = _forced_ctc_jamo_confidences(logprobs, ref_ids, _blank_id, confusable_ids=conf_ids)
            coverage_ratio = float(np.mean([c > MISSING_THR for c in confs_ref])) if len(confs_ref) > 0 else 0.0

        # duration sanity
        expected_sec = max(0.0, len(mapped_tokens)) * (DUR_PER_JAMO_MS / 1000.0)
        dur_ok = True
        if expected_sec >= 0.35:  # đừng áp quy tắc quá chặt với câu cực ngắn
            dur_ok = (voiced_sec >= DUR_TOL_LOW * expected_sec) and (voiced_sec <= DUR_TOL_HIGH * expected_sec)

        # quyết định gate
        gate_pass = (
            (voiced_ratio >= VOICED_MIN) and
            (avg_top1_nonblank >= TOP1_NONBLANK_MIN) and
            (coverage_ratio >= COVERAGE_MIN) and
            (repetition_run <= REP_RUN_MAX) and
            dur_ok and
            ((cer < cer_threshold) or (mean_neglogp < loss_char_threshold))
        )

        return {
            "gate_pass": bool(gate_pass),
            "cer": float(cer),
            "mean_neglogp": float(mean_neglogp),
            "voiced_ratio": float(voiced_ratio),
            "avg_top1_nonblank": float(avg_top1_nonblank),
            "repetition_run": float(repetition_run),
            "voiced_sec": float(voiced_sec),
            "expected_sec": float(expected_sec),
            "coverage_ratio": float(coverage_ratio),
        }

def ctc_gate_global(wav_path: str, ref_text_nfd: str, **kwargs) -> Dict[str, float]:
    wav, sr = torchaudio.load(wav_path)
    return ctc_gate_global_from_waveform(wav, sr, ref_text_nfd, **kwargs)
