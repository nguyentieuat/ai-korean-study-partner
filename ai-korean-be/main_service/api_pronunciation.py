import json
from flask import Blueprint, jsonify, request
import requests
from pathlib import Path
from datetime import datetime

pronunciation_bp = Blueprint('pronunciation', __name__)

pronunciation_data = {}

for level in range(1, 6):
    filename = f'data/level{level}_pronunciation_data.json'
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        pronunciation_data[level] = data

# mfa_service_url = 'http://localhost:5002'
mfa_service_url = 'http://13.213.18.34:5002'

@pronunciation_bp.route('/api/pronunciation/<level>', methods=['GET'])
def get_data_by_level(level):
    try:
        level_int = int(level)
    except ValueError:
        return jsonify({"error": "Cấp độ không hợp lệ"}), 400
    
    data = pronunciation_data.get(level_int)
    if not data:
        return jsonify({"error": "Không tìm thấy cấp độ"}), 404
    return jsonify({"level": level, "items": data})


UPLOAD_FOLDER = Path("static/uploads/pronun")

@pronunciation_bp.route('/api/pronunciation/evaluate', methods=['POST'])
def pronunciation_evaluate():
    try:
        text = request.form.get("text")
        audio_file = request.files.get("audio")

        if not text or not audio_file:
            return jsonify({"error": "Thiếu text hoặc audio"}), 400

        # --- Gửi sang mfa_service ---
        try:
            response = requests.post(
                f'{mfa_service_url}/api/evaluate',
                data={'text': text},
                files = {
                            'audio': (audio_file.filename, audio_file.stream, audio_file.content_type)
                        }
            )
            result = response.json()
            print(f"[INFO] MFA service response: {result}")
        except Exception as e:
            return jsonify({"error": f"Lỗi gọi mfa_service: {e}"}), 500
        

        # --- Lưu file người dùng ---
        today_str = datetime.now().strftime("%Y-%m-%d")
        user_folder = UPLOAD_FOLDER / today_str
        user_folder.mkdir(parents=True, exist_ok=True)

        audio_filename = f"{datetime.now().strftime('%H%M%S')}_{text}.wav"
        audio_path = user_folder / audio_filename

        audio_file.seek(0)  # Đặt lại vị trí file pointer
        audio_file.save(audio_path)

        # Ghi log nếu muốn
        log_dir = Path("log")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "low_score_log.jsonl"
        with open(log_path, "a", encoding="utf-8") as log_file:
             with open("log/low_score_log.jsonl", "a", encoding="utf-8") as log_file:
                log_file.write(json.dumps({
                    "text": text,
                    "audioFile": str(audio_path),
                    "time": datetime.now().isoformat()
                }, ensure_ascii=False) + "\n")

        return jsonify({
                "text": text,
                "score": result.get("avg_score", 0),
                "detail": result.get("phonemes", []),  # gửi nguyên mảng phonemes về
                "audioFile": str(audio_path)
            })
    except ValueError:
        return jsonify({"error": "Cấp độ không hợp lệ"}), 400
