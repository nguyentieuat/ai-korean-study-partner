import os
import requests
import difflib
from dotenv import load_dotenv; load_dotenv()
import traceback

# === CONFIG ===
API_KEY = os.getenv("OPENROUTER_API_KEY") 

def generate_reply(prompt_history: list) -> str:
    try:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "HTTP-Referer": "https://yourapp.com",  # Thay bằng domain dự án
            "X-Title": "AI Korean Speaking Partner"
        }

        # prompt_history là danh sách messages dạng dict: [{"role": "user", "content": "..."}, ...]
        data = {
            "model": "gpt-4o-mini",  # hoặc "meta-llama/llama-3-8b-instruct:free"
            "messages": prompt_history,
            "temperature": 0.7,
            "max_tokens": 500
        }

        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            print(f"Exception: {e}")
            traceback.print_exc()
            return "죄송합니다. 답변을 생성하지 못했습니다."

    except Exception as e:
        print(f"Exception: {e}")
        traceback.print_exc()
        return "죄송합니다. 문제가 발생했습니다."
