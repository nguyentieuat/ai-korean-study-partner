from flask import Blueprint, jsonify, request
import requests
import os, uuid
import uuid

topikgenerate_bp = Blueprint('topikgenerate', __name__)


# tqg_service_url = 'http://localhost:5003'
# chat_service_url = 'http://localhost:5001'
tqg_service_url = 'http://54.255.225.190:5003'
chat_service_url = 'http://13.250.33.57:5001'

@topikgenerate_bp.route('/api/generate_question', methods=['POST'])
def topik_generate_question():
    data = request.get_json()
    level = data.get("level", "").lower()
    category = data.get("category")
    cau = data.get("cau")
    
    audio_url = None  # ✅ khởi tạo trước

    try:
        # Gọi TQG service
        response = requests.post(
            f'{tqg_service_url}/api/generate_question',
            json={"level": level, "category": category, "cau": cau}
        )
        result = response.json()
        question_text = result.get("question", {}).get("question", "")
        if result["type"].startswith("Nghe"):
            # Gọi TTS service
            response_tts = requests.post(
                f'{chat_service_url}/api/generate_tts',
                json={"text": question_text}
            )
        
            if response_tts.status_code == 200:
                audio_bytes = response_tts.content
                filename = f"question_{uuid.uuid4().hex}.mp3"
                folder = os.path.join("static", "tts")
                os.makedirs(folder, exist_ok=True)
                path = os.path.join(folder, filename)
                with open(path, "wb") as f:
                    f.write(audio_bytes)
                audio_url = f"/static/tts/{filename}"
            else:
                print("❌ Lỗi khi gọi TTS:", response_tts.text)

        result["question_audio_url"] = audio_url
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Lỗi gọi chat_service: {e}"}), 500
