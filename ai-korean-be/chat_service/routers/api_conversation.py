# api_conversation.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Response
from pydantic import BaseModel
from typing import List, Any, Optional
from io import BytesIO
from werkzeug.datastructures import FileStorage
import traceback

# Giữ nguyên import logic từ project của bạn
from speech.stt import transcribe_audio
from speech.tts import text_to_speech
from utils.utils import process_conversation

router = APIRouter(prefix="/api", tags=["conversation"])

# ----- Pydantic models -----
class ChatRequest(BaseModel):
    user_input: str
    history: List[Any] = []
    check_grammar_flag: bool = False
    conversation_id: Optional[str] = None                

class TTSRequest(BaseModel):
    text: str
    voice: Optional[int] = 1  # 1 = nam (default), 0 = nữ

# ----- Endpoints -----
@router.post("/generate_chat")
def generate_chat(payload: ChatRequest):
    try:
        result = process_conversation(
            payload.user_input,
            payload.history,
            payload.check_grammar_flag
        )

        # Lưu lịch sử hội thoại
        payload.history.append({"role": "user", "content": payload.user_input})
        payload.history.append({"role": "assistant", "content": result.get("reply", "")})
        conversation_id = payload.conversation_id if hasattr(payload, 'conversation_id') else None
        if conversation_id:
            save_conversation(conversation_id, payload.history)
        
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate_tts")
def generate_tts(payload: TTSRequest):
    try:
        audio_bytes = text_to_speech(payload.text or "", voice=payload.voice)
        if not audio_bytes:
            raise HTTPException(status_code=500, detail="Tạo TTS thất bại")
        return Response(content=audio_bytes, media_type="audio/mpeg",
                        headers={"Content-Disposition": 'inline; filename="tts.mp3"'})
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/transcribe")
async def transcribe(
    file_storage: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None),
):
    up = file_storage or file
    if up is None or (up.filename or "") == "":
        raise HTTPException(status_code=400, detail="Thiếu file upload")

    try:
        # Chuyển UploadFile (FastAPI) -> FileStorage (Werkzeug) để tái dùng transcribe_audio cũ
        content = await up.read()
        fs = FileStorage(
            stream=BytesIO(content),
            filename=up.filename,
            content_type=up.content_type or "application/octet-stream",
        )
        text = transcribe_audio(fs)
        return {"transcript": text}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

import os, json
def save_conversation(conversation_id, history):

    folder = os.path.join("data", "history", "conversation")
    os.makedirs(folder, exist_ok=True)

    filepath = os.path.join(folder, f"{conversation_id}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)