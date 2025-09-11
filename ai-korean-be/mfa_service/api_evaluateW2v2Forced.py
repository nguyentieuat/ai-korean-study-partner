# api_evaluateW2v2Forced.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydub import AudioSegment
from pathlib import Path
from tempfile import TemporaryDirectory
import uuid
import traceback
import torch

# W2V2 forced scoring (không cần TG user)
from w2v2_forced_aligner.w2v2_forced_scoring import (
    extract_features,
    score_pronunciation_forced,
    init_w2v2,
    estimate_char_confidences_from_energy,   # NEW
)

# Map text -> (ref_wav, ref_textgrid)  (TextGrid ref dùng làm priors nếu có)
from data.first.mapping_data_first import get_paths_by_text

# G2P để lấy ref_chunks (label + phones)
try:
    from utils.utils import load_mfa_dict, text_to_phonemes_mfa
except Exception:
    def load_mfa_dict(_): return {}
    def text_to_phonemes_mfa(text, mfa_dict, return_by_word=False):
        if return_by_word:
            return [(text, [])], []
        return []

MFA_DICT_PATH = "./korean_mfa.dict"

router = APIRouter(prefix="/api", tags=["evaluate-W2V2-Forced"])

@router.on_event("startup")
async def _warmup_models():
    init_w2v2()  # idempotent

@router.post("/evaluate-w2v2-forced")
def evaluate_pronunciation_forced(
    text: str = Form(...),
    audio: UploadFile = File(...),
):
    """
    W2v2Forced:
      - KHÔNG cần TextGrid của user
      - Có thể dùng TextGrid của REF (nếu tồn tại) làm duration priors
      - Gate im lặng: char_confidences được ước lượng theo năng lượng ⇒ im lặng ⇒ score=0 + advice 'thiếu'
    """
    try:
        with TemporaryDirectory(prefix="w2v2_forced_") as tmpdir:
            session_dir = Path(tmpdir)
            wav_path = session_dir / "sample.wav"

            # 1) Chuẩn hoá WAV 16k mono
            raw_bytes = audio.file.read()
            if not raw_bytes:
                raise HTTPException(status_code=400, detail="File audio rỗng")
            tmp_in = session_dir / f"in_{uuid.uuid4().hex}"
            tmp_in.write_bytes(raw_bytes)
            try:
                snd = AudioSegment.from_file(tmp_in)
                snd = snd.set_channels(1).set_frame_rate(16000)
                snd.export(wav_path, format="wav")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Lỗi xử lý audio: {e}")

            # 2) Lấy total duration (đồng thời warm model lần nữa nếu cần)
            feats, sr_user, total_sec = extract_features(str(wav_path))

            # vectors_user tối thiểu (chỉ để truyền duration/schema)
            mean_vec = feats.mean(dim=0) if hasattr(feats, "mean") else torch.zeros(1)
            vectors_user = [{
                "phoneme": "_",
                "vector": mean_vec.detach().float(),
                "start": 0.0,
                "end": float(total_sec),
            }]

            # 3) Ước lượng char_confidences theo năng lượng (RMS)
            char_confs = estimate_char_confidences_from_energy(str(wav_path), text)

            # 4) Optional: TextGrid REF (nếu có) → priors
            ref_tg_path = None
            ref_paths = get_paths_by_text(text)
            if ref_paths:
                _ref_wav, _ref_tg = ref_paths
                if _ref_tg and Path(_ref_tg).exists():
                    ref_tg_path = _ref_tg

            # 5) G2P → ref_chunks
            try:
                mfa_dict = load_mfa_dict(MFA_DICT_PATH)
            except Exception:
                mfa_dict = {}
            ref_chunks, notes = text_to_phonemes_mfa(text, mfa_dict, return_by_word=True)

            # 6) Chấm điểm (chunk-aware, priors nếu có, GATE im lặng bằng char_confs)
            scoring = score_pronunciation_forced(
                vectors_user=vectors_user,
                reference_text=text,
                ref_chunks=ref_chunks,
                ref_textgrid_path=str(ref_tg_path) if ref_tg_path else None,
                char_confidences=char_confs,                # <<<<<<<<<< key: im lặng -> 0
                missing_threshold=0.05,     # ↑
                min_score_floor=0.0,        # giữ 0.0: thiếu => 0 điểm
                advice_threshold=0.80,
            )

            return JSONResponse({
                "notes": notes,
                "text": text,
                **scoring
            })

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
