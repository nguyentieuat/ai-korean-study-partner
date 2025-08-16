from flask import Blueprint, Response, jsonify, request
import os, uuid, json
import uuid
from io import BytesIO
from speech.tts import text_to_speech
from werkzeug.datastructures import FileStorage
from speech.transcribe import transcribe_audio
from utils.utils import process_conversation, save_audio_upload

conversation_bp = Blueprint('conversation', __name__)

@conversation_bp.route('/api/generate_chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data['user_input']  # key phải khớp với json gửi từ main_service
    history = data.get('history', [])
    check_grammar_flag = data.get('check_grammar_flag', False)

    try:
        result = process_conversation(user_input, history, check_grammar_flag)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@conversation_bp.route('/api/generate_tts', methods=['POST'])
def handle_voice():
    data = request.get_json()
    text = data.get("text", "")
    try:
        # convert reply to TTS
        audio_bytes = text_to_speech(text)
        if audio_bytes is None:
            return "Tạo TTS thất bại", 500

        # Trả bytes trực tiếp
        return Response(audio_bytes, mimetype="audio/mpeg")
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@conversation_bp.route('/api/transcribe', methods=['POST'])
def transcribe_endpoint():
    file_storage = request.files['file_storage']

    if file_storage.filename == '':
        return jsonify({'error': 'Tên file rỗng'}), 400

    try:
        # Chuyển âm thanh thành văn bản
        transcript = transcribe_audio(file_storage)
        return jsonify({"transcript": transcript})
    except Exception as e:
        return jsonify({'error': str(e)}), 500