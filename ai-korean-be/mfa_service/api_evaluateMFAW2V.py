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
from run_mfa_align_single import run_mfa_align_single
from tempfile import TemporaryDirectory
from ml_based_prototype_scoring.prototype_scoring  import extract_features, load_alignment, extract_phoneme_vectors, score_pronunciation
from data_model_first.mapping_model_first import get_paths_by_text


evaluateMFAW2V_bp = Blueprint('evaluateMFAW2V', __name__)

@evaluateMFAW2V_bp.route("/api/evaluate", methods=["POST"])
def evaluate_pronunciation():
    text = request.form.get("text")
    audio = request.files.get("audio")

    if not text or not audio:
        return jsonify({"error": "Thiếu text hoặc audio"}), 400

    session_id = str(uuid.uuid4())
    session_folder = Path("temp_data") / session_id
    session_folder.mkdir()

    lab_path = session_folder / "sample.lab"
    wav_path = session_folder / "sample.wav"
    align_out = session_folder / "align"

    # Ghi text vào .lab
    with open(lab_path, "w", encoding="utf-8") as f:
        f.write(text)

    # Lưu audio thành wav 16kHz mono
    try:
        audio.save(wav_path)
        sound = AudioSegment.from_file(wav_path)
        sound = sound.set_channels(1).set_frame_rate(16000)
        sound.export(wav_path, format="wav")
    except Exception as e:
        return jsonify({"error": f"Lỗi audio: {e}"}), 500

    # Chạy MFA align
    try:
        # --- Align trực tiếp ---
        print(wav_path, lab_path, align_out)
        run_mfa_align_single(wav_path, lab_path, align_out)
    except Exception as e:
        return jsonify({"error": f"MFA lỗi: {e}"}), 500

    tg_path = align_out / "sample.TextGrid"
    if not tg_path.exists():
        return jsonify({"error": "Không tìm thấy file TextGrid"}), 500

    # --- Extract features và alignment cho user ---
    features_user, sr_user, total_sec_user = extract_features(str(wav_path))
    phones_user = load_alignment(str(tg_path))
    vectors_user = extract_phoneme_vectors(features_user, sr_user, phones_user, total_sec_user)

    # --- Extract features và alignment cho ref ---
    # file audio tham khảo và TextGrid tương ứng cho câu text đó
    resultPath = get_paths_by_text(text)
    if resultPath:
        ref_wav_path, ref_tg_path = resultPath
    else:
        return jsonify({"error": "Không tìm thấy file TextGrid Model"}), 500

    features_ref, sr_ref, total_sec_ref = extract_features(str(ref_wav_path))
    phones_ref = load_alignment(str(ref_tg_path))
    vectors_ref = extract_phoneme_vectors(features_ref, sr_ref, phones_ref, total_sec_ref)

    # --- So sánh và tính điểm ---
    scoring_result = score_pronunciation(vectors_user, vectors_ref, phones_user)

    # Trả kết quả JSON gồm điểm tổng và chi tiết từng phoneme
    return jsonify({
        "text": text,
        "avg_score": scoring_result["avg_score"],   # điểm tổng
        "details": scoring_result["details"],       # danh sách phoneme với điểm và màu
    })


   