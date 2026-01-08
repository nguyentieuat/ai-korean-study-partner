# api_conversation.py
import os, uuid, base64, traceback
from typing import List, Any, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Request
from pydantic import BaseModel
from utils.utils import text_to_speech

router = APIRouter(prefix="/api", tags=["conversation"])

CHAT_SERVICE_URL = os.getenv("CHAT_SERVICE_URL", "http://127.0.0.1:5001").rstrip("/")
ASR_SERVICE_URL  = os.getenv("ASR_SERVICE_URL",  "http://127.0.0.1:5005").rstrip("/")
TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://127.0.0.1:5004").rstrip("/")

# ---------- Models ----------
class ChatReq(BaseModel):
    message: str
    conversation_history: List[Any] = []
    conversation_id: Optional[str] = None
    # voice có thể được gửi kèm khi bạn muốn TTS cho text mode trong tương lai
    voice: Optional[int] = 1  # 1=nam, 0=nữ (không dùng ở chatting hiện tại)

class TalkReq(BaseModel):
    transcript: str
    history: List[Any] = []
    conversation_id: Optional[str] = None
    voice: Optional[int] = 1  # 1=nam (mặc định), 0=nữ

# ---------- Endpoints ----------
@router.post("/korean-speaking-chatting")
async def chatting(req: Request, body: ChatReq):
    conv_id = body.conversation_id or str(uuid.uuid4())
    http = req.app.state.http
    try:
        resp = await http.post(
            f"{CHAT_SERVICE_URL}/api/generate_chat",
            json={
                "user_input": body.message,
                "history": body.conversation_history,
                "check_grammar_flag": True,
                "conversation_id": conv_id,
            },
        )
        if resp.is_error:
            raise HTTPException(resp.status_code, f"chat_service: {resp.text[:500]}")
        result = resp.json()
        result["conversation_id"] = conv_id
        return result
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Lỗi gọi chat_service: {e}")

@router.post("/korean-speaking-talking")
async def talking(req: Request, body: TalkReq):
    conv_id = body.conversation_id or str(uuid.uuid4())
    http = req.app.state.http
    try:
        # 1) Gọi chat_service để lấy reply text
        r_chat = await http.post(
            f"{CHAT_SERVICE_URL}/api/generate_chat",
            json={
                "user_input": body.transcript,
                "history": body.history,
                "check_grammar_flag": True,
                "conversation_id": conv_id,
            },
        )
        if r_chat.is_error:
            raise HTTPException(r_chat.status_code, f"chat_service: {r_chat.text[:500]}")
        result = r_chat.json()
        reply_text = result.get("reply", "") or ""

        # 2) Gọi TTS (forward voice: 1=nam, 0=nữ)
        tts_b64 = text_to_speech(
            text=reply_text,
            voice=body.voice if body.voice is not None else 1,
            rate="-10%",
            pitch=None,
            volume=None,
            timeout=30,
        )

        # Dựng data URL từ base64 (mặc định audio/mpeg)
        tts_data_url = f"data:audio/mpeg;base64,{tts_b64}" if tts_b64 else None

        return {
            "ai_reply_tts": reply_text,
            "tts_audio_url": tts_data_url,        # FE play trực tiếp (data URL)
            "highlighted": result.get("highlighted", ""),
            "conversation_id": conv_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))

@router.post("/transcribe")
async def transcribe(req: Request, audio: UploadFile = File(...)):
    """
    Nhận audio, gọi ASR, trả transcript + audio gốc dưới dạng data URL (base64)
    """
    http = req.app.state.http
    if not audio.filename:
        raise HTTPException(400, "Tên file rỗng")

    try:
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(400, "File audio rỗng")

        r = await http.post(
            f"{ASR_SERVICE_URL}/api/stt/",
            files={"file": (audio.filename, audio_bytes, audio.content_type or "application/octet-stream")},
            data={"language": "ko"},
        )
        if r.is_error:
            raise HTTPException(r.status_code, f"ASR: {r.text[:500]}")
        payload = r.json()
        transcript = (payload.get("text") or "").strip()

        b64 = base64.b64encode(audio_bytes).decode("ascii")
        data_url = f"data:{audio.content_type or 'audio/mpeg'};base64,{b64}"

        return {
            "transcript": transcript,
            "language": payload.get("language"),
            "model": payload.get("model"),
            "duration": payload.get("duration"),
            "audio_url_goc": data_url,
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))
