import os
import requests
import difflib
import re
from dotenv import load_dotenv

# === CONFIG ===
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY") 
MODEL = "openai/gpt-3.5-turbo"  # Hoặc dùng llama3 như: "meta-llama/llama-3-70b-instruct"

# === HÀM CHÍNH ===
def check_grammar(text: str) -> dict:
    print(f"[INFO] Checking grammar for text: {text}")
    prompt = (
        "아래 문장에서 문법, 철자 오류가 있다면 수정해 주세요. "
        "수정한 문장을 먼저 출력하고, 어떤 부분이 왜 수정되었는지 간단히 설명해 주세요.\n"
        f"문장: {text}"
    )

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "당신은 한국어 교사입니다."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
    result = response.json()
    print(f"[INFO] Grammar check response: {result}")

    if "choices" not in result:
        return {"error": result.get("error", "Unknown error")}

    content = result["choices"][0]["message"]["content"]
    lines = content.strip().split('\n')
    corrected = re.sub(r"^.*?:\s*", "", lines[0]).strip()
    explanation = "\n".join(lines[1:]).strip()

    if normalize(corrected) == normalize(text):
        return {
            "original": text,
            "corrected": "",
            "highlighted": "",
            "explanation": "문법이나 철자 오류가 없습니다."
        }

    highlighted = highlight_changes(text, corrected)

    return {
        "original": text,
        "corrected": corrected,
        "highlighted": highlighted,
        "explanation": explanation
    }


# === HÀM PHỤ: Tô màu phần thay đổi ===
def highlight_changes(original, corrected):
    """
    Highlight các cụm từ khác biệt (không bao gồm dấu chấm cuối câu).
    """
    # Loại bỏ dấu chấm kết câu tạm thời để không detect
    end_punct_pattern = r"[.?!…]$"
    original_core = re.sub(end_punct_pattern, "", original)
    corrected_core = re.sub(end_punct_pattern, "", corrected)

    # Dùng SequenceMatcher để tìm đoạn khác nhau
    sm = difflib.SequenceMatcher(None, original_core, corrected_core)
    result = ""
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            result += corrected_core[j1:j2]
        elif tag == 'replace' or tag == 'insert':
            result += f'<span class="highlight">{corrected_core[j1:j2]}</span>'
        # Xóa không cần thêm gì

    # Lấy lại dấu kết câu (nếu có) từ corrected
    end_punct = corrected[-1] if re.search(end_punct_pattern, corrected) else ""

    return result + end_punct


def normalize(text):
    return re.sub(r'[.!?…]+$', '', text.strip())