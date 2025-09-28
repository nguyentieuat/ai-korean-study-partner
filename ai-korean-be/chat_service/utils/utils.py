import subprocess
import os
import sys
import uuid
from dialogue_engine.dialog_generator import generate_reply
from grammar_checker.grammar_checker import check_grammar

def convert_webm_to_mp3(webm_path):
    mp3_path = webm_path.replace(".webm", ".mp3")
    subprocess.run([
        "ffmpeg", "-i", webm_path,
        "-vn", "-ar", "44100", "-ac", "2", "-b:a", "192k",
        mp3_path
    ])
    return mp3_path


def save_audio_upload(audio_bytes, file_type):
    try:
        upload_dir = os.path.join("static", "uploads", file_type)
        os.makedirs(upload_dir, exist_ok=True)

        audio_filename = f"{file_type}_{uuid.uuid4().hex}.webm"
        audio_path = os.path.join(upload_dir, audio_filename)

        if not audio_bytes:
            raise ValueError("File audio rỗng hoặc không đọc được dữ liệu.")

        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        audio_url_goc = f"/static/uploads/{file_type}/{audio_filename}"
        return audio_url_goc

    except Exception as e:
        print(f"[Save Audio Error] {str(e)}", file=sys.stderr)
        raise RuntimeError("Không thể lưu file audio.") from e


def process_conversation(user_input, history, check_grammar_flag=True):
    if not user_input or user_input.strip() == "":
        raise ValueError("user_input trống")

    # 1. Grammar check nếu là text
    if check_grammar_flag:
        grammar_result = check_grammar(user_input)
        print(f"[INFO] Grammar check result: {grammar_result}")
        if 'error' in grammar_result:
            raise Exception("Grammar check failed: " + grammar_result['error'])
        corrected_text = grammar_result['corrected']
    else:
        grammar_result = {
            'original': user_input,
            'corrected': user_input,
            'highlighted': None,
            'explanation': None
        }
        corrected_text = user_input

    # 2. Tạo prompt cho AI
    prompt_history = [{"role": "system", "content": (
        "너는 한국어 회화 파트너야. 오직 한국어로만 대답해. "
        "어떤 경우에도 영어 단어나 문장을 사용하지 마."
    )
    }]
    for item in history[-5:]:  # chỉ lấy 5 lượt gần nhất
        role = item.get("role", "").lower()
        msg = item.get("message")
        if msg:
            prompt_history.append({"role": role, "content": msg})

    prompt_history.append({"role": "user", "content": corrected_text})

    # 3. Generate reply
    ai_reply = generate_reply(prompt_history)
    if not ai_reply:
        raise Exception("Reply generation failed")

    return {
        "original": user_input,
        "corrected": corrected_text,
        "highlighted": grammar_result.get("highlighted"),
        "explanation": grammar_result.get("explanation"),
        "reply": ai_reply,
        "updated_history": prompt_history  # có thể sử dụng lại
    }

