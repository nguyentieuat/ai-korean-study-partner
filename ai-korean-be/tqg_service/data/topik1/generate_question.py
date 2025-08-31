import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

# Load JSON prompt
with open("data/prompt_template/topik_i_prompt.json", "r", encoding="utf-8") as f:
    prompts = json.load(f)["topik_I"]

# ----- Custom Exceptions -----
class PromptNotFoundError(Exception):
    """Không tìm thấy prompt phù hợp"""
    pass

class AIServiceError(Exception):
    """Lỗi từ dịch vụ AI"""
    pass

# ----- Helper -----
def get_prompt_by_cau(cau: int, subtype: str = None, lang: str = "ko"):
    """
    Tìm prompt phù hợp theo số câu (cau).
    - cau: số câu (int)
    - subtype: nếu có (ví dụ action/place/transport/object cho 1~4)
    - lang: "ko" hoặc "vi" để chọn ngôn ngữ prompt
    """
    for p in prompts:
        start, end = map(int, p["range"].split("-"))
        if start <= int(cau) <= end:
            # Nếu có subtypes (chỉ áp dụng cho 1~4)
            if "subtypes" in p and subtype:
                for st in p["subtypes"]:
                    if st["type"] == subtype:
                        return st[f"prompt_{lang}"]
                raise PromptNotFoundError(f"Không tìm thấy subtype {subtype} trong range {p['range']}")
            else:
                return p[f"prompt_{lang}"]
    raise PromptNotFoundError(f"Không tìm thấy prompt cho câu {cau}")

# ----- Service -----
def generate_topik_question(cau: int, subtype: str = None, lang: str = "ko"):
    """
    Sinh câu TOPIK I dựa trên số câu.
    Args:
        cau (int): số câu
        subtype (str): loại câu hỏi (nếu có, dùng cho 1-4)
        lang (str): "ko" hoặc "vi" (prompt tiếng Hàn hoặc tiếng Việt)
    """
    question_prompt = get_prompt_by_cau(cau, subtype, lang)

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "한국어 선생님이시군요. TOPIK 1급 형식으로 문제를 만들어 보세요."},
            {"role": "user", "content": question_prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }

    try:
        resp = requests.post(url, headers=headers, json=data, timeout=30)
    except requests.RequestException as e:
        raise AIServiceError(f"Lỗi kết nối đến OpenRouter: {e}")

    if resp.status_code != 200:
        raise AIServiceError(f"AI call lỗi {resp.status_code}: {resp.text}")

    resp_json = resp.json()
    try:
        ai_output = resp_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise AIServiceError(f"AI response không hợp lệ: {e}")

    try:
        question_json = json.loads(ai_output)
    except json.JSONDecodeError:
        raise AIServiceError(f"Không parse được JSON từ AI: {ai_output}")

    return {
        "cau": cau,
        "subtype": subtype,
        "prompt_lang": lang,
        "question": question_json
    }
