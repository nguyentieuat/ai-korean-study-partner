# api_evaluateW2v2Forced.py
from __future__ import annotations

import io
import re
import unicodedata
import traceback
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
import torch
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydub import AudioSegment

# === Core scorers (encoder + forced scoring; CTC được import nội bộ trong module này) ===
from w2v2_forced_aligner.w2v2_forced_scoring import (
    init_w2v2,
    extract_features_from_waveform,                         # no I/O
    estimate_char_confidences_from_energy_from_waveform,    # no I/O
    score_pronunciation_forced,                             # chấm forced; gọi CTC gate qua ctc_gate.*
)

# === CTC init riêng: sau khi split, init_ctc nằm ở ctc_gate.py ===
from w2v2_forced_aligner.ctc_gate import init_ctc

# === REF mapping: text -> (ref_wav, ref_textgrid) (nếu có, dùng priors) ===
from data.first.mapping_data_first import get_paths_by_text

# === G2P (MFA) để tạo ref_chunks ===
try:
    from utils.utils import load_mfa_dict, text_to_phonemes_mfa
except Exception:
    def load_mfa_dict(_): return {}
    def text_to_phonemes_mfa(text, mfa_dict, return_by_word=False):
        if return_by_word:
            return [(text, [])], []
        return []

# =========================
# Config & Router
# =========================
MFA_DICT_PATH = "./korean_mfa.dict"
TARGET_SR = 16000

# Thresholds & behavior (tinh chỉnh tuỳ UX)
CTC_LOSS_CHAR_THRESHOLD = 6.0     # ↑ từ 3.8 → nới nhiều
CTC_CER_THRESHOLD = 0.88          # ↑ từ 0.75
CTC_MIX_MODE = "max"              # "max" → chọn cái tự tin hơn giữa energy/CTC
MISSING_THRESHOLD = 0.02          # ↓ từ 0.05 → ít cờ thiếu âm
MIN_SCORE_FLOOR = 0.45            # ↑ từ 0.25 → sàn điểm cao hơn
ADVICE_THRESHOLD = 0.70           # ↓ từ 0.80 → ít gợi ý thiếu

router = APIRouter(prefix="/api", tags=["evaluate-W2V2-Forced"])

# =========================
# Lifecycle
# =========================
@router.on_event("startup")
async def _warmup_models() -> None:
    """
    Khởi tạo encoder (W2V2) và CTC để giảm latency cho request đầu tiên.
    """
    init_w2v2()  # idempotent
    init_ctc()   # idempotent

# =========================
# Helpers
# =========================
def _normalize_korean_text(s: str) -> str:
    """
    Giữ Hangul + khoảng trắng, loại bỏ ký tự ngoài (dấu câu/ký tự Latin...)
    nhằm ổn định số ký tự/syllable khi chia spans.
    """
    s = unicodedata.normalize("NFC", s).strip()
    s = re.sub(r"[^\uAC00-\uD7A3\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _decode_upload_to_tensor(upload: UploadFile) -> Tuple[torch.Tensor, int]:
    """
    Decode UploadFile -> mono float32 waveform [-1, 1], sr = TARGET_SR.
    Hỗ trợ 16-bit, 24-bit, 32-bit PCM và float.
    """
    raw = upload.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="File audio rỗng")

    seg = AudioSegment.from_file(io.BytesIO(raw))
    seg = seg.set_channels(1).set_frame_rate(TARGET_SR)

    sample_width = seg.sample_width
    frame_count = len(seg.get_array_of_samples())

    # Lấy bytes gốc để tự chuyển theo sample_width
    pcm_bytes = seg.raw_data

    if sample_width == 2:
        # 16-bit signed PCM
        x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    elif sample_width == 3:
        # 24-bit signed PCM -> mở rộng lên int32 với sign-extend
        a = np.frombuffer(pcm_bytes, dtype=np.uint8).astype(np.uint32)
        a = a.reshape(-1, 3)
        # ghép thành 32-bit little-endian: b0 + (b1<<8) + (b2<<16)
        val = (a[:, 0] | (a[:, 1] << 8) | (a[:, 2] << 16)).astype(np.int32)
        # sign-extend 24-bit: nếu bit 23 set, trừ 1<<24
        neg = val & 0x00800000
        val = val - (neg << 1)
        x = val.astype(np.float32) / 8388608.0  # 2^23

    elif sample_width == 4:
        # Có thể là 32-bit PCM signed hoặc float32 tuỳ codec. Pydub thường đưa về PCM.
        # Thử đoán: nếu biên độ tuyệt đối >1.5 là PCM int32; nếu |x|<=1.0 là float32.
        arr = np.frombuffer(pcm_bytes, dtype=np.int32)
        # Heuristic: nếu nhiều mẫu vượt 1e6 -> coi là int32 PCM
        if (np.abs(arr) > 1_000_000).mean() > 0.01:
            x = arr.astype(np.float32) / 2147483648.0
        else:
            # Thử float32
            try:
                x = np.frombuffer(pcm_bytes, dtype=np.float32)
                if not np.isfinite(x).all():
                    raise ValueError
            except Exception:
                # fallback: coi là int32
                x = arr.astype(np.float32) / 2147483648.0
    else:
        # Fallback: pydub’s array_of_samples (có thể là 'h'), rồi normalize
        x = np.array(seg.get_array_of_samples()).astype(np.float32)
        maxv = float(np.max(np.abs(x)) or 1.0)
        x = x / maxv

    # Chuẩn hoá gain vừa phải để tránh quá nhỏ
    peak = float(np.max(np.abs(x)) or 0.0)
    if peak < 1e-3:
        # audio quá nhỏ -> khuếch đại nhẹ
        x = x * (1.0 / max(peak, 1e-6))
    x = np.clip(x, -1.0, 1.0)

    wav = torch.from_numpy(x).unsqueeze(0)  # [1, N]
    return wav, TARGET_SR

# =========================
# Endpoint
# =========================
@router.post("/evaluate-w2v2-forced")
def evaluate_pronunciation_forced(
    text: str = Form(...),
    audio: UploadFile = File(...),
):
    """
    Đánh giá phát âm KHÔNG cần TextGrid của user (forced mode).
    - Gate nói bừa bằng CTC (posterior & CER).
    - Trộn posterior CTC với năng lượng để phát hiện thiếu âm.
    - Nếu có TextGrid tham chiếu, dùng làm duration priors.
    """
    try:
        # 1) Decode audio -> waveform tensor (mono, 16k)
        wav, sr = _decode_upload_to_tensor(audio)

        # 2) Extract features (để biết tổng thời lượng) — không ghi file
        feats, _, total_sec = extract_features_from_waveform(wav, sr)

        # Vectors_user tối thiểu: chỉ dùng để truyền duration/schema cho forced mode
        mean_vec = feats.mean(dim=0) if hasattr(feats, "mean") else torch.zeros(1)
        vectors_user = [{
            "phoneme": "_",
            "vector": mean_vec.detach().float(),
            "start": 0.0,
            "end": float(total_sec),
        }]

        # 3) Chuẩn hoá text cho năng lượng & G2P; vẫn trả text gốc ra JSON
        text_norm = _normalize_korean_text(text)

        # 4) Ước lượng char confidences theo năng lượng (không I/O)
        char_confs_energy = estimate_char_confidences_from_energy_from_waveform(
            wav, sr, text_norm
        )

        # 5) Optional: TextGrid REF → priors
        ref_tg_path = None
        ref_paths = get_paths_by_text(text) or get_paths_by_text(text_norm)
        if ref_paths:
            _ref_wav, _ref_tg = ref_paths
            if _ref_tg and Path(_ref_tg).exists():
                ref_tg_path = _ref_tg

        # 6) G2P (MFA) → ref_chunks
        try:
            mfa_dict = load_mfa_dict(MFA_DICT_PATH)
        except Exception:
            mfa_dict = {}
        ref_chunks, notes = text_to_phonemes_mfa(text_norm, mfa_dict, return_by_word=True)

        print({
            "dbg_peak": float(wav.abs().max().item()),
            "dbg_rms": float((wav.pow(2).mean().sqrt()).item()),
            "dbg_len_sec": float(wav.shape[-1]) / TARGET_SR,
            "dbg_text_norm": text_norm
            })

        # 7) Chấm điểm (forced + CTC gate)
        scoring = score_pronunciation_forced(
            wav_path=None,
            waveform=wav, waveform_sr=sr,
            vectors_user=vectors_user,
            reference_text=text_norm,
            ref_chunks=ref_chunks,
            ref_textgrid_path=str(ref_tg_path) if ref_tg_path else None,
            char_confidences=char_confs_energy,
            missing_threshold=MISSING_THRESHOLD,
            min_score_floor=MIN_SCORE_FLOOR,
            advice_threshold=ADVICE_THRESHOLD,
            use_ctc_gate=True,
            ctc_loss_char_threshold=CTC_LOSS_CHAR_THRESHOLD,
            ctc_cer_threshold=CTC_CER_THRESHOLD,
            ctc_mix_mode=CTC_MIX_MODE,
        )

        return JSONResponse({
            "notes": notes,      # ghi chú từ G2P (nếu có)
            "text": text,        # trả câu gốc người dùng nhập
            **scoring
        })

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
