import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

# Load JSON prompt
with open("data/prompt_template/topik_i_prompt.json", "r", encoding="utf-8") as f:
    prompts = json.load(f)

# ----- Custom Exceptions -----
class PromptNotFoundError(Exception):
    """Không tìm thấy prompt phù hợp"""
    pass

class AIServiceError(Exception):
    """Lỗi từ dịch vụ AI"""
    pass

# ----- Helper -----
def get_prompt_by_cau(danh_muc: str, cau: str):
    """
    Tìm prompt phù hợp theo danh mục và số câu.
    Hỗ trợ khoảng câu như "49-50,53-56,59-62,65-68".
    """
    for p in prompts:
        if p["danh_muc"] != danh_muc:
            continue

        cau_list = []
        for part in p["cau"].split(","):
            if "-" in part:
                start, end = map(int, part.split("-"))
                cau_list.extend(range(start, end + 1))
            else:
                cau_list.append(int(part))

        if int(cau) in cau_list:
            return p
    return None

# ----- Service -----
def generate_topik_question(danh_muc: str, cau: str):
    """
    Sinh câu TOPIK I dựa trên danh mục và số câu.
    Raises:
        PromptNotFoundError
        AIServiceError
    Returns:
        dict: JSON câu hỏi TOPIK
    """
    question_prompt = get_prompt_by_cau(danh_muc, cau)
    if not question_prompt:
        raise PromptNotFoundError(f"Không tìm thấy prompt cho {danh_muc} - câu {cau}")

    # Call OpenRouter
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "한국어 선생님이시군요. TOPIK 1급 형식으로 문제를 만들어 보세요."},
            {"role": "user", "content": question_prompt["prompt"]},
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
        "type": question_prompt['danh_muc'],  # dùng danh_muc có sẵn trong JSON
        "title": question_prompt['title'],  # dùng title có sẵn trong JSON
        "question": question_json
    }
