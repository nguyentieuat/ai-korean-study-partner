# api_evaluateASRneedleman.py
import os
import uuid
import traceback
import tempfile
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydub import AudioSegment

from g2p_needleman_wunsch_alignment.g2p_needleman_alignment import evaluate
from data.first.mapping_data_first import get_paths_by_text  

router = APIRouter(prefix="/api", tags=["evaluate-needleman"])

@router.post("/evaluate-needleman")
async def evaluate_pronunciation(
    text: str = Form(...),
    audio: UploadFile = File(...),
):
    if not text or not audio:
        raise HTTPException(400, "Thiếu text hoặc audio")

    # Thư mục tạm riêng cho mỗi request (giống tinh thần session folder trước đây)
    with tempfile.TemporaryDirectory(prefix="temp_data_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        raw_path = tmpdir_path / f"in_{uuid.uuid4().hex}"
        wav_path = tmpdir_path / "sample.wav"

        # 1) Lưu file upload ra đĩa (để pydub đọc)
        try:
            contents = await audio.read()
            raw_path.write_bytes(contents)
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(500, f"Lỗi lưu audio tạm: {e}")

        # 2) Chuẩn hoá về mono 16k WAV
        try:
            # pydub cần ffmpeg trong PATH
            snd = AudioSegment.from_file(raw_path)
            snd = snd.set_channels(1).set_frame_rate(16000)
            snd.export(wav_path, format="wav")
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(500, f"Lỗi chuyển đổi audio: {e}")

        # 3) Lấy TextGrid tham chiếu theo text
        try:
            resultPath = get_paths_by_text(text)
            if not resultPath:
                raise HTTPException(500, "Không tìm thấy file TextGrid Model")
            _, textgrid_file = resultPath
            textgrid_file = str(textgrid_file)
            if not os.path.isfile(textgrid_file):
                raise HTTPException(500, f"TextGrid không tồn tại: {textgrid_file}")
        except HTTPException:
            raise
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(500, f"Lỗi tìm TextGrid: {e}")

        # 4) Gọi evaluate(...)
        try:
            output = evaluate(
                audio_file=str(wav_path),
                reference_text=text,
                textgrid_file=textgrid_file,
            )
            # output được giả định là dict (giống Flask)
            return {"text": text, **output}
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(500, f"Lỗi evaluate: {e}")
