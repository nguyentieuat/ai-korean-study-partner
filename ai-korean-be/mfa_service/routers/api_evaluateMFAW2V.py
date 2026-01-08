# api_evaluateMFAW2V.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydub import AudioSegment
from pathlib import Path
from tempfile import TemporaryDirectory
import uuid
import traceback

# MFA align (1 file)
from run_mfa_align_single import run_mfa_align_single

# Vec2Vec scoring
from ml_based_prototype_scoring.prototype_scoring import (
    init_w2v2,                # <<< NEW: import init
    extract_features,
    load_alignment,
    extract_phoneme_vectors,
    score_pronunciation,
)

# Map text -> (ref_wav, ref_textgrid)
from data.first.mapping_data_first import get_paths_by_text

# === NEW: G2P để lấy ref_chunks (label + phones) ===
try:
    from utils.utils import load_mfa_dict, text_to_phonemes_mfa
except Exception:
    def load_mfa_dict(_): return {}
    def text_to_phonemes_mfa(text, mfa_dict, return_by_word=False):
        if return_by_word:
            return [(text, [])], []
        return []

MFA_DICT_PATH = "./korean_mfa.dict"

router = APIRouter(prefix="/api", tags=["evaluate-MFA-W2V"])

# --- Load & warmup model ngay khi app khởi chạy ---
@router.on_event("startup")
def _startup_load_w2v2():
    init_w2v2()

@router.post("/evaluate-mfa-w2v")
def evaluate_pronunciation(
    text: str = Form(...),
    audio: UploadFile = File(...),
):
    """
    Nhận text + audio (multipart/form-data), chuẩn hoá WAV 16k mono,
    chạy MFA align để ra TextGrid, trích vector và so sánh với mẫu (vec2vec).
    Trả về: { text, score, detail, (by_chunk nếu có) }
    """
    try:
        # Tạo thư mục tạm cho phiên này
        with TemporaryDirectory(prefix="mfa_eval_") as tmpdir:
            session_dir = Path(tmpdir)
            lab_path = session_dir / "sample.lab"
            wav_path = session_dir / "sample.wav"
            align_out = session_dir / "align"
            align_out.mkdir(exist_ok=True)

            # 1) Lưu text -> .lab
            lab_path.write_text(text, encoding="utf-8")

            # 2) Lưu audio và chuẩn hoá WAV 16k mono
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

            # 3) Chạy MFA align -> sample.TextGrid
            try:
                run_mfa_align_single(wav_path, lab_path, align_out)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"MFA lỗi: {e}")

            tg_path = align_out / "sample.TextGrid"
            if not tg_path.exists():
                raise HTTPException(status_code=500, detail="Không tìm thấy file TextGrid sau khi align")

            # 4) Extract features & alignment cho user
            features_user, sr_user, tot_user = extract_features(str(wav_path))
            phones_user = load_alignment(str(tg_path))
            vectors_user = extract_phoneme_vectors(features_user, sr_user, phones_user, tot_user)

            # 5) Lấy ref (wav + TextGrid) theo text
            ref_paths = get_paths_by_text(text)
            if not ref_paths:
                raise HTTPException(status_code=404, detail="Không tìm thấy audio/TextGrid mẫu cho câu này")
            ref_wav_path, ref_tg_path = ref_paths

            features_ref, sr_ref, tot_ref = extract_features(str(ref_wav_path))
            phones_ref = load_alignment(str(ref_tg_path))
            vectors_ref = extract_phoneme_vectors(features_ref, sr_ref, phones_ref, tot_ref)

            # 6) NEW: Lấy ref_chunks từ G2P cho text tham chiếu (để gán nhãn chunk & gộp by_chunk)
            try:
                mfa_dict = load_mfa_dict(MFA_DICT_PATH)
            except Exception:
                mfa_dict = {}
            ref_chunks, notes = text_to_phonemes_mfa(text, mfa_dict, return_by_word=True)

            # 7) So sánh vec2vec + chunk-aware
            scoring = score_pronunciation(
                vectors_user, vectors_ref, phones_user,
                ref_chunks=ref_chunks
            )

            return JSONResponse({
                "note": notes,
                "text": text,
                **scoring
            })
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
