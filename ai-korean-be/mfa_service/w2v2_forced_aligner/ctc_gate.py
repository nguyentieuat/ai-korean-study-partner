
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
import unicodedata as ud
import torch.nn as nn

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

import unicodedata as ud
import torch.nn as nn

def _to_nfd_jamo_only(s: str) -> str:
    # Chuẩn hoá mạnh → lọc đúng Jamo NFD
    s = ud.normalize("NFKD", s)
    return "".join(ch for ch in s if 0x1100 <= ord(ch) <= 0x11FF)

def _is_vocab_nfd(vocab: Dict[str, int]) -> bool:
    # Heuristic: nếu có ᄋ (U+110B) thì là NFD; nếu chỉ có ㅇ (U+3147) là compatibility
    return ("ᄋ" in vocab) or ("ᄀ" in vocab)

def _map_ref_to_vocab_space(ref_nfd: str, vocab: Dict[str, int]) -> str:
    # Nếu tokenizer dùng Compatibility Jamo, map NFD → Compatibility để target vào đúng vocab
    if _is_vocab_nfd(vocab):
        return ref_nfd
    out = []
    for ch in ref_nfd:
        cp = ord(ch)
        if 0x1100 <= cp <= 0x1112:  # choseong
            base_map = {
                0x1100: 0x3131, 0x1102: 0x3134, 0x1103: 0x3137, 0x1105: 0x3139,
                0x1106: 0x3141, 0x1107: 0x3142, 0x1109: 0x3145, 0x110B: 0x3147,
                0x110C: 0x3148, 0x110E: 0x314A, 0x110F: 0x314B, 0x1110: 0x314C,
                0x1111: 0x314D, 0x1112: 0x314E,
            }
            out.append(chr(base_map.get(cp, cp)))
        elif 0x1161 <= cp <= 0x1175:  # vowels
            out.append(chr(cp - 0x1161 + 0x314F))
        elif 0x11A8 <= cp <= 0x11C2:  # jong
            jong_map = {
                0x11A8: 0x3131, 0x11AB: 0x3134, 0x11AE: 0x3137, 0x11AF: 0x3139,
                0x11B7: 0x3141, 0x11B8: 0x3142, 0x11BA: 0x3145, 0x11BC: 0x3147,
                0x11BD: 0x3148, 0x11BE: 0x314A, 0x11BF: 0x314B, 0x11C0: 0x314C,
                0x11C1: 0x314D, 0x11C2: 0x314E,
            }
            out.append(chr(jong_map.get(cp, cp)))
        # ký tự khác bỏ qua
    return "".join(out)

def _cer(a: str, b: str) -> float:
    if not b:
        return 1.0 if a else 0.0
    # Levenshtein
    if a == b:
        return 0.0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j-1] + 1
            dele = prev[j] + 1
            rep = prev[j-1] + (ca != cb)
            cur.append(min(ins, dele, rep))
        prev = cur
    return prev[-1] / max(1, len(b))

def _cer_skeleton(nfd_a: str, nfd_b: str) -> float:
    # CER chỉ trên bộ khung CV (bỏ jong 0x11A8..0x11C2) để khoan dung mất jong nhẹ
    a = "".join(ch for ch in nfd_a if (0x1100 <= ord(ch) <= 0x1112) or (0x1161 <= ord(ch) <= 0x1175))
    b = "".join(ch for ch in nfd_b if (0x1100 <= ord(ch) <= 0x1112) or (0x1161 <= ord(ch) <= 0x1175))
    return _cer(a, b)

def _ctc_ref_nll_per_char(logprobs: torch.Tensor, vocab: Dict[str,int],
                          ref_text_nfd: str, blank_id: int) -> float:
    """
    Tính CTCLoss (NLL) của câu ref trực tiếp trên logprobs, chia theo số ký tự target.
    Không phụ thuộc decode; rất hợp để gate.
    """
    tgt_str = _map_ref_to_vocab_space(_to_nfd_jamo_only(ref_text_nfd), vocab)
    ref_ids = [vocab[ch] for ch in tgt_str if ch in vocab]
    if len(ref_ids) == 0 or logprobs.shape[0] == 0:
        return 10.0  # cực xấu
    # PyTorch CTCLoss expects (T, N, C)
    logp = logprobs.unsqueeze(1)  # (T,1,V)
    input_lengths = torch.tensor([logp.shape[0]], dtype=torch.long, device=logp.device)
    target_lengths = torch.tensor([len(ref_ids)], dtype=torch.long, device=logp.device)
    targets = torch.tensor(ref_ids, dtype=torch.long, device=logp.device)
    ctc_loss = nn.CTCLoss(blank=blank_id, reduction="sum", zero_infinity=True)
    loss = ctc_loss(logp, targets, input_lengths, target_lengths)  # scalar
    return float(loss.item() / max(1, len(ref_ids)))

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
import unicodedata as ud

def _to_nfd_jamo_only(s: str) -> str:
    # Chuẩn hoá về NFKD để tách Compatibility Jamo thành NFD Jamo, rồi lọc vùng U+1100..U+11FF
    s = ud.normalize("NFKD", s)
    return "".join(ch for ch in s if 0x1100 <= ord(ch) <= 0x11FF)

def _is_vocab_nfd(vocab: Dict[str, int]) -> bool:
    # Nếu vocab có 'ᄋ' (U+110B) là NFD; nếu không nhưng có 'ㅇ' (U+3147) thì là Compatibility
    return ("ᄋ" in vocab) or ("ᄀ" in vocab)  # heuristic nhẹ cho NFD

def _map_ref_to_vocab_space(ref_nfd: str, vocab: Dict[str, int]) -> str:
    """
    Nếu vocab là Compatibility, chuyển từng Jamo NFD → Compatibility Jamo 1-1.
    Nếu vocab là NFD, giữ nguyên.
    """
    if _is_vocab_nfd(vocab):
        return ref_nfd  # vocab NFD → giữ nguyên
    # vocab Compatibility → map từng Jamo
    out = []
    for ch in ref_nfd:
        cp = ord(ch)
        # Leading consonants U+1100..U+1112 → U+3131..U+314E (approx)
        if 0x1100 <= cp <= 0x1112:
            # Bảng ánh xạ tối thiểu cho phụ âm đầu
            base_map = {
                0x1100: 0x3131,  # ᄀ → ㄱ
                0x1102: 0x3134,  # ᄂ → ㄴ
                0x1103: 0x3137,  # ᄃ → ㄷ
                0x1105: 0x3139,  # ᄅ → ㄹ
                0x1106: 0x3141,  # ᄆ → ㅁ
                0x1107: 0x3142,  # ᄇ → ㅂ
                0x1109: 0x3145,  # ᄉ → ㅅ
                0x110B: 0x3147,  # ᄋ → ㅇ
                0x110C: 0x3148,  # ᄌ → ㅈ
                0x110E: 0x314A,  # ᄎ → ㅊ
                0x110F: 0x314B,  # ᄏ → ㅋ
                0x1110: 0x314C,  # ᄐ → ㅌ
                0x1111: 0x314D,  # ᄑ → ㅍ
                0x1112: 0x314E,  # ᄒ → ㅎ
            }
            out.append(chr(base_map.get(cp, cp)))
        # Vowels U+1161..U+1175 → U+314F..U+3163
        elif 0x1161 <= cp <= 0x1175:
            out.append(chr(cp - 0x1161 + 0x314F))
        # Final consonants U+11A8..U+11C2 → Compatibility tương ứng (xấp xỉ)
        elif 0x11A8 <= cp <= 0x11C2:
            jong_map = {
                0x11A8: 0x3131,  # ㄱ
                0x11AB: 0x3134,  # ㄴ
                0x11AE: 0x3137,  # ㄷ
                0x11AF: 0x3139,  # ㄹ
                0x11B7: 0x3141,  # ㅁ
                0x11B8: 0x3142,  # ㅂ
                0x11BA: 0x3145,  # ㅅ
                0x11BC: 0x3147,  # ㅇ
                0x11BD: 0x3148,  # ㅈ
                0x11BE: 0x314A,  # ㅊ
                0x11BF: 0x314B,  # ㅋ
                0x11C0: 0x314C,  # ㅌ
                0x11C1: 0x314D,  # ㅍ
                0x11C2: 0x314E,  # ㅎ
            }
            out.append(chr(jong_map.get(cp, cp)))
        else:
            # Bỏ ký tự ngoài Jamo
            pass
    return "".join(out)

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
    + Chuẩn hoá/mapping ref → đúng vocab.
    + Fallback no-VAD nếu VAD cắt rỗng.
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

        # --- VAD crop (lần 1) ---
        vslice = _blank_posterior_vad(logprobs_full, _blank_id, thresh=blank_thr, pad_ms=pad_ms)
        logprobs = logprobs_full[vslice]  # [T_crop, V]
        T_full = logprobs_full.shape[0]
        sec_per_frame = total_sec / max(1, T_full)
        crop_start_sec = vslice.start * sec_per_frame

        # --- map ref → vocab space ---
        vocab = _processor.tokenizer.get_vocab()
        ref_clean = _to_nfd_jamo_only(ref_text_nfd)
        ref_text_for_vocab = _map_ref_to_vocab_space(ref_clean, vocab)

        ref_tokens_all = [t for t in list(ref_text_for_vocab) if t]
        ref_ids = [vocab[t] for t in ref_tokens_all if t in vocab]
        aligned_priors: List[Optional[Tuple[float,float]]] = []
        if priors_sec is not None:
            # căn chỉnh priors theo các token còn giữ lại
            j = 0
            for ch in list(ref_clean):
                if 0x1100 <= ord(ch) <= 0x11FF:
                    mapped = _map_ref_to_vocab_space(ch, vocab)
                    if j < len(priors_sec) and mapped in ref_tokens_all:
                        aligned_priors.append(priors_sec[j])
                    elif j < len(priors_sec):
                        aligned_priors.append(None)
                    j += 1

        # --- Fallback nếu VAD rỗng hoặc ref_ids rỗng ---
        if logprobs.shape[0] == 0 or len(ref_ids) == 0:
            # thử lại không VAD
            logprobs = logprobs_full
            crop_start_sec = 0.0
            if len(ref_ids) == 0:
                # nếu vẫn 0 sau khi map → trả zero theo độ dài ref (để FE không vỡ)
                return [0.0] * max(1, len(ref_tokens_all))

        # Build confusable map (Jamo only)
        conf_ids = None
        if use_confusables:
            conf_ids = {}
            for t, alts in CONFUSABLES.items():
                if t in vocab:
                    conf_ids[vocab[t]] = [vocab[a] for a in alts if a in vocab]

        # Build bias for priors (nếu có)
        bias_ext = None
        if aligned_priors and any(p is not None for p in aligned_priors):
            bias_ext = _build_bias_ext_for_priors(
                logprobs.shape[0], _expand_ref_with_blanks(ref_ids, _blank_id), _blank_id,
                aligned_priors, sec_per_frame, crop_start_sec,
                out_penalty=PRIOR_OUT_PENALTY, edge_pad_sec=PRIOR_EDGE_PAD_SEC
            )

        confs = _forced_ctc_jamo_confidences(logprobs, ref_ids, _blank_id, confusable_ids=conf_ids, bias_ext=bias_ext)

        # pad để giữ đúng độ dài ref_tokens_all cho FE
        if len(confs) < len(ref_tokens_all):
            confs = confs + [0.0] * (len(ref_tokens_all) - len(confs))

        # debug nhẹ để bạn thấy nguyên nhân nếu 0
        try:
            print(f"[ctc_char_conf] T_full={T_full}, T_crop={logprobs.shape[0]}, "
                  f"len(ref_tokens_all)={len(ref_tokens_all)}, len(ref_ids)={len(ref_ids)}, "
                  f"mean_conf={float(np.mean(confs)) if len(confs)>0 else -1}")
        except Exception:
            pass

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
                                  cer_threshold: float = 0.65,
                                  loss_char_threshold: float = 2.0,
                                  ref_nll_threshold: float = float(os.getenv("W2V_CTC_REF_NLL_THR", "2.2")),
                                  short_syll_cer_bonus: float = 0.05) -> Dict[str, float]:
    """
    Gate “lai”: pass nếu (A) CER_NFD + mean_neglogp đạt, hoặc (B) CTC-NLL/ref_char thấp.
    Với câu ngắn (<=5 âm tiết) nới nhẹ CER một chút để tránh mất âm đầu/cuối do VAD.
    """
    _init()
    wav = _ensure_sr(waveform, sr)
    with torch.inference_mode():
        inputs = _processor(wav.squeeze(0), sampling_rate=TARGET_SR, return_tensors="pt", padding=False)
        input_values = inputs.input_values.to(DEVICE)
        logits = _model(input_values).logits[0]
        logprobs_full = _log_softmax(logits, temp=LOGIT_TEMP)

        sl = _blank_posterior_vad(logprobs_full, _blank_id, thresh=BLANK_VAD_THRESH, pad_ms=VAD_PAD_MS)
        logprobs = logprobs_full[sl]

        vocab = _processor.tokenizer.get_vocab()
        id2tok = {v: k for k, v in vocab.items()}

        # Best-path chỉ để logging / CER tham khảo
        hyp = _best_path_decode(logprobs, id2tok)      # theo vocab thực tế
        hyp_nfd = _to_nfd_jamo_only(hyp)              # đưa về NFD để CER công bằng
        ref_nfd = _to_nfd_jamo_only(ref_text_nfd)

        cer = _cer(hyp_nfd, ref_nfd)
        cer_skel = _cer_skeleton(hyp_nfd, ref_nfd)

        # mean_neglogp như cũ (thô theo trung bình posterior)
        ref_ids_for_mean = [vocab[ch] for ch in _map_ref_to_vocab_space(ref_nfd, vocab) if ch in vocab]
        if len(ref_ids_for_mean) == 0 or logprobs.shape[0] == 0:
            mean_neglogp = 10.0
        else:
            tok_post = logprobs.exp().mean(0)  # [V]
            pick = tok_post[ref_ids_for_mean].clamp_min(1e-6)
            mean_neglogp = float((-pick.log()).mean().item())

        # NLL theo CTC trực tiếp trên câu tham chiếu (độc lập decode)
        ref_nll_pc = _ctc_ref_nll_per_char(logprobs, vocab, ref_nfd, _blank_id)

        # Nới nhẹ CER cho câu ngắn (<=5 âm tiết ~ 15 jamo)
        n_syl_est = max(1, len(ref_nfd) // 3)
        cer_thr_eff = cer_threshold + (short_syll_cer_bonus if n_syl_est <= 5 else 0.0)

        # Gate logic: đạt 1 trong 3 nhánh là pass
        gate_by_pair = (cer < cer_thr_eff and mean_neglogp < loss_char_threshold)
        gate_by_nll = (ref_nll_pc < ref_nll_threshold)
        gate_by_short = (n_syl_est <= 2 and cer_skel < 0.6)  # cực ngắn thì chỉ cần khung CV khớp

        gate_pass = bool(gate_by_pair or gate_by_nll or gate_by_short)

        # Log cho dễ debug
        print(f"Hyp: {hyp} | Hyp_NFD: {hyp_nfd}")

        return {
            "cer": float(cer),
            "cer_skeleton": float(cer_skel),
            "mean_neglogp": float(mean_neglogp),
            "ref_nll_per_char": float(ref_nll_pc),
            "gate_by_pair": bool(gate_by_pair),
            "gate_by_nll": bool(gate_by_nll),
            "gate_by_short": bool(gate_by_short),
            "gate_pass": gate_pass,
        }

def ctc_gate_global(wav_path: str, ref_text_nfd: str, **kwargs) -> Dict[str, float]:
    wav, sr = torchaudio.load(wav_path)
    return ctc_gate_global_from_waveform(wav, sr, ref_text_nfd, **kwargs)