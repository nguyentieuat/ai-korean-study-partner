# dialogue_engine/dialog_generator.py
import os, json, time, re
import requests
from dotenv import load_dotenv

load_dotenv()

# ===== Local LLM-Service (ưu tiên) =====
DIALOGUE_LOCAL_ENABLED = os.getenv("DIALOGUE_LOCAL_ENABLED", "true").strip().lower() != "false"
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://localhost:5006").rstrip("/")
LOCAL_TIMEOUT_CONN = int(os.getenv("LOCAL_TIMEOUT_CONN", "5"))
LOCAL_TIMEOUT_READ = int(os.getenv("LOCAL_TIMEOUT_READ", "320"))

# ===== OpenRouter fallback =====
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Ưu tiên các model OS/phủ rộng; OpenAI để cuối
MODEL_CANDIDATES = [
    os.getenv("DIALOGUE_MODEL", "").strip() or "gpt-4o-mini",
    "anthropic/claude-3.5-sonnet",
    "meta-llama/llama-3-8b-instruct",
    "meta-llama/llama-3-70b-instruct",
    "google/gemma-2-9b-it",
    "mistralai/mixtral-8x7b-instruct",
    "gpt-4o-mini",
    "openai/gpt-3.5-turbo",
]

def _extract_text_from_choices(result: dict) -> str:
    try:
        choices = result.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    except Exception:
        pass
    return ""

# ---------- LOCAL CALL (ưu tiên) ----------
def _call_local_chat(prompt_history: list) -> str:
    """
    Gọi llm-service local /chat
    Body: { messages: [...], temperature?: float }
    Trả string content; rỗng nếu lỗi để cho fallback.
    """
    if not DIALOGUE_LOCAL_ENABLED:
        return ""
    url = f"{LLM_SERVICE_URL}/chat"
    payload = {
        "messages": prompt_history,
        "temperature": 0.7,
    }
    try:
        r = requests.post(url, json=payload, timeout=(LOCAL_TIMEOUT_CONN, LOCAL_TIMEOUT_READ))
        if r.status_code // 100 != 2:
            return ""
        data = r.json()
        return (data.get("content") or "").strip()
    except requests.RequestException:
        return ""
    except Exception:
        return ""

# ---------- OPENROUTER FALLBACK ----------
def _call_openrouter_chat(prompt_history: list) -> str:
    if not OPENROUTER_API_KEY:
        return ""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    base_payload = {
        "messages": prompt_history,
        "temperature": 0.7,
        "max_tokens": 500,
    }

    for model in MODEL_CANDIDATES:
        if not model:
            continue
        data = dict(base_payload)
        data["model"] = model

        for attempt in range(2):
            try:
                resp = requests.post(
                    OPENROUTER_URL, headers=headers, json=data,
                    timeout=(10, 30)
                )
                if resp.status_code // 100 != 2:
                    if resp.status_code in (401, 403, 429, 503):
                        break
                    break

                result = resp.json()
                content = _extract_text_from_choices(result)
                if content:
                    return content.strip() or "죄송합니다. 빈 응답이 생성되었습니다."
                else:
                    break

            except requests.Timeout:
                if attempt == 0:
                    time.sleep(0.5)
                    continue
                break
            except Exception:
                break

    return ""

# ---------- PUBLIC ----------
def generate_reply(prompt_history: list) -> str:
    """
    prompt_history: [{"role": "system|user|assistant", "content": "..."}]
    """
    # 1) thử local trước
    out = _call_local_chat(prompt_history)
    if out:
        return out

    # 2) fallback OpenRouter
    out = _call_openrouter_chat(prompt_history)
    if out:
        return out

    # 3) hết đường
    if DIALOGUE_LOCAL_ENABLED:
        return "죄송합니다. 현재 로컬 LLM 서비스 또는 백업 모델에 연결할 수 없습니다. 잠시 후 다시 시도해 주세요."
    else:
        return "죄송합니다. 현재 모델에 연결할 수 없습니다. 잠시 후 다시 시도해 주세요."
