from flask import Blueprint, jsonify, request
import requests
import os, uuid, json
import uuid
from io import BytesIO
from werkzeug.datastructures import FileStorage
import traceback
from utils.utils import save_audio_upload

conversation_bp = Blueprint('conversation', __name__)

# chat_service_url = 'http://localhost:5001'
chat_service_url = 'http://13.250.33.57:5001'

def save_conversation(conversation_id, history):
    folder = os.path.join("cooperate", "conversation")
    os.makedirs(folder, exist_ok=True)

    filepath = os.path.join(folder, f"{conversation_id}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

@conversation_bp.route('/api/korean-speaking-chatting', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data['message']
    history = data.get('conversation_history', [])
    
    # --- Gửi sang chat_service ---
    try:
        response = requests.post(
            f'{chat_service_url}/api/generate_chat',
                json={
                    "user_input": user_input,
                    "history": history,
                    "check_grammar_flag": True
                }
            )
        result = response.json()

        # Lưu lịch sử hội thoại
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": result.get("reply", "")})
        conversation_id = data.get("conversation_id")
        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        save_conversation(conversation_id, history)
        return jsonify(result)
    except Exception as e:
        print(f"Exception: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Lỗi gọi chat_service: {e}"}), 500

   
@conversation_bp.route('/api/korean-speaking-talking', methods=['POST'])
def handle_voice():
    data = request.get_json()
    transcript = data.get("transcript", "")
    history = data.get("history", [])

    try:
        response_chat = requests.post(
            f'{chat_service_url}/api/generate_chat',
                json={
                    "user_input": transcript,
                    "history": history,
                    "check_grammar_flag": True
                }
            )
        result = response_chat.json()

         # Lưu lịch sử hội thoại
        history.append({"role": "user", "content": transcript})
        history.append({"role": "assistant", "content": result.get("reply", "")})
        conversation_id = data.get("conversation_id")
        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        save_conversation(conversation_id, history)

        # convert reply to TTS
        response_tts = requests.post(
            f'{chat_service_url}/api/generate_tts',
                json={"text": result['reply']}
            )
        print("[INFO] TTS service response:", response_tts.json())
        
        if response_tts.status_code == 200:
            # Lấy mảng byte MP3 từ response
            audio_bytes = response_tts.content

            # Tạo file ngẫu nhiên
            filename = f"reply_{uuid.uuid4().hex}.mp3"
            folder = os.path.join("static", "tts")
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, filename)

            # Lưu file
            with open(path, "wb") as f:
                f.write(audio_bytes)

            audio_url = f"/static/tts/{filename}"
        else:
            audio_url = None

        result["tts_audio_url"] = audio_url

        return jsonify({
            'ai_reply_tts': result['reply'],
            'tts_audio_url': result["tts_audio_url"]
        })
    except Exception as e:
        print(f"Exception: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
@conversation_bp.route('/api/transcribe', methods=['POST'])
def transcribe_endpoint():
    if 'audio' not in request.files:
        return jsonify({'error': 'Không có file audio'}), 400

    audio_file = request.files['audio']

    if audio_file.filename == '':
        return jsonify({'error': 'Tên file rỗng'}), 400

    try:
        audio_bytes = audio_file.read()
        audio_url_goc = save_audio_upload(audio_bytes, "chat")

        # Tạo lại stream để dùng cho FileStorage
        audio_stream = BytesIO(audio_bytes)

        file_storage = FileStorage(
            stream=audio_stream,
            filename=audio_file.filename,
            content_type=audio_file.content_type
        )

        # Chuyển âm thanh thành văn bản
        response = requests.post(
            f'{chat_service_url}/api/transcribe',
                files = {
                            "file_storage": (file_storage.filename, file_storage.stream, file_storage.content_type)
                        }
            )
        transcript = response.json().get("transcript", "")
        return jsonify({
            'transcript': transcript,
            'audio_url_goc': audio_url_goc
        })
    except Exception as e:
        print(f"Exception: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500