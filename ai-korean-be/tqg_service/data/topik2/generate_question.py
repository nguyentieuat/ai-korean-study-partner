import json
import os
import requests
from dotenv import load_dotenv
from fastapi import HTTPException

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

# Load JSON prompt
with open("data/prompt_template/topik_i_prompt.json", "r", encoding="utf-8") as f:
    prompts = json.load(f)

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

def generate_topik_question(danh_muc: str, cau: str):
    """
    Sinh câu TOPIK I dựa trên danh mục và số câu.
    """
    question_prompt = get_prompt_by_cau(danh_muc, cau)
    if not question_prompt:
        raise HTTPException(status_code=404, detail="Không tìm thấy prompt phù hợp")

    # Prompt AI tổng quát
    prompt_text_AI = f"""
당신은 한국어 교사입니다. Level TOPIK II 문제 형식으로 문제를 하나 생성하세요.

- 문제 유형: {question_prompt['dang_kr']}
- 정확한 답은 하나만 있습니다.
- 다른 선택지는 완전히 틀리거나 적절하지 않아야 합니다.
- JSON 형식으로 출력하세요: {question_prompt['prompt']}
- 질문 내용과 선택지는 모두 한국어로 작성하세요.
- 한국어 외 언어(예: 베트남어, 로마자)는 포함하지 마세요.
"""

    # Call OpenRouter
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "mistralai/mistral-7b-instruct:free",
        "messages": [
            {"role": "system", "content": prompt_text_AI},
            {"role": "user", "content": question_prompt["prompt"]},
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }

    resp = requests.post(url, headers=headers, json=data)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    resp_json = resp.json()
    try:
        ai_output = resp_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise HTTPException(status_code=500, detail="AI response không hợp lệ")

    try:
        question_json = json.loads(ai_output)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Không parse được JSON: {ai_output}")

    return question_json
