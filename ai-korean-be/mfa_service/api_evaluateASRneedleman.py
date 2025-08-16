import json
from flask import Blueprint, jsonify, request
import os
import json
import shutil
import uuid
import subprocess
from pathlib import Path
from pydub import AudioSegment
from textgrid import TextGrid
from g2p_needleman_wunsch_alignment.g2p_needleman_alignment import evaluate

evaluateASRneedleman_bp = Blueprint('evaluateASRneedleman', __name__)

@evaluateASRneedleman_bp.route("/api/evaluate", methods=["POST"])
def evaluate_pronunciation():
    text = request.form.get("text")
    audio = request.files.get("audio")

    if not text or not audio:
        return jsonify({"error": "Thiếu text hoặc audio"}), 400

    # Tạo folder tạm cho session
    session_id = str(uuid.uuid4())
    session_folder = Path("temp_data") / session_id
    session_folder.mkdir(parents=True, exist_ok=True)
    wav_path = session_folder / "sample.wav"

    # Lưu audio và chuẩn hóa
    try:
        audio.save(wav_path)
        sound = AudioSegment.from_file(wav_path)
        sound = sound.set_channels(1).set_frame_rate(16000)
        sound.export(wav_path, format="wav")
    except Exception as e:
        return jsonify({"error": f"Lỗi audio: {e}"}), 500

    # Lấy đường dẫn TextGrid tham chiếu (có thể dùng hàm mapping của bạn)
    from data_model_first.mapping_model_first import get_paths_by_text
    resultPath = get_paths_by_text(text)

    if not resultPath:
        return jsonify({"error": "Không tìm thấy file TextGrid Model"}), 500
    _, textgrid_file = resultPath

    try:
        output = evaluate(
            audio_file=str(wav_path),
            reference_text=text,
            textgrid_file=str(textgrid_file)
        )
        return jsonify({"text": text, **output})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

   