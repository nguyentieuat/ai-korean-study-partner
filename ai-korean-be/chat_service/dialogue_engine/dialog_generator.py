# dialogue_engine/dialog_generator.py
import os, json, time, traceback, re
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Ưu tiên các model open-source/phủ rộng, để cuối là OpenAI (dễ bị chặn theo khu vực)
MODEL_CANDIDATES = [
    os.getenv("DIALOGUE_MODEL", "").strip() or "gpt-4o-mini",
    "anthropic/claude-3.5-sonnet"
    "meta-llama/llama-3-8b-instruct",
    "meta-llama/llama-3-70b-instruct",
    "google/gemma-2-9b-it",
    "mistralai/mixtral-8x7b-instruct",
    "gpt-4o-mini",
    "openai/gpt-3.5-turbo",  # để cuối cùng
]

def _extract_text_from_choices(result: dict) -> str:
    """
    Trích nội dung trả về từ nhiều định dạng khác nhau.
    """
    try:
        choices = result.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Một số model trả dạng list-of-parts
            return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    except Exception:
        pass
    return ""

def generate_reply(prompt_history: list) -> str:
    """
    prompt_history: danh sách messages [{"role": "system/user/assistant", "content": "..."}]
    Trả về: string nội dung assistant.
    """
    # Không có API key → trả thông báo nhẹ, tránh crash BE
    if not OPENROUTER_API_KEY:
        return "죄송합니다. (설정 오류) OPENROUTER_API_KEY가 없어 답변을 생성할 수 없습니다."

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        # "HTTP-Referer": "https://yourapp.com",
        # "X-Title": "AI Korean Speaking Partner",
    }

    # Base payload (các trường khác giữ nguyên)
    base_payload = {
        "messages": prompt_history,
        "temperature": 0.7,
        "max_tokens": 500,
    }

    # Thử lần lượt từng model, mỗi model retry nhẹ 1 lần khi timeout
    for model in MODEL_CANDIDATES:
        if not model:
            continue

        data = dict(base_payload)
        data["model"] = model

        for attempt in range(2):  # tối đa 2 lần cho mỗi model (retry timeout)
            try:
                resp = requests.post(
                    OPENROUTER_URL,
                    headers=headers,
                    json=data,
                    timeout=(10, 30),  # 10s connect, 30s read
                )

                # HTTP lỗi → quyết định chuyển model
                if resp.status_code // 100 != 2:
                    # Với 401/403/429/503 thường do quota/region/rate-limit → thử model tiếp theo
                    if resp.status_code in (401, 403, 429, 503):
                        break
                    # Lỗi khác cũng chuyển model (ghi log ngắn)
                    # print(f"[{model}] HTTP {resp.status_code}: {resp.text[:200]}")
                    break

                # Parse JSON
                result = resp.json()
                content = _extract_text_from_choices(result)
                if content:
                    return content.strip() or "죄송합니다. 빈 응답이 생성되었습니다."
                else:
                    # Không có choices/content → chuyển model
                    break

            except requests.Timeout:
                # Cho phép retry đúng 1 lần khi timeout
                if attempt == 0:
                    time.sleep(0.5)
                    continue
                break
            except Exception:
                # Lỗi mạng/parse khác → chuyển model
                # traceback.print_exc()
                break

    # Nếu tất cả model đều fail (region/quota/network…)
    return "죄송합니다. 현재 지역/네트워크 제한으로 답변을 생성하지 못했습니다. 잠시 후 다시 시도해 주세요."
