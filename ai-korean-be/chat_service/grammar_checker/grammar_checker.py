# grammar_checker/grammar_checker.py
import os, json, time
import requests, difflib, re
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

# Thứ tự fallback: ưu tiên model open-source/hỗ trợ rộng rãi để tránh lỗi region
MODEL_CANDIDATES = [
    os.getenv("GRAMMAR_MODEL", "").strip() or "gpt-4o-mini",
    "anthropic/claude-3.5-sonnet",
    "meta-llama/llama-3-8b-instruct",
    "meta-llama/llama-3-70b-instruct",
    "google/gemma-2-9b-it",
    "mistralai/mixtral-8x7b-instruct",
    "openai/gpt-3.5-turbo",   # để cuối cùng (có thể bị chặn theo khu vực)
]

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ---------- helpers ----------
def _normalize(text: str) -> str:
    return re.sub(r'[.!?…]+$', '', (text or "").strip())

def _highlight_changes(original: str, corrected: str) -> str:
    end_punct_pattern = r"[.?!…]$"
    original_core = re.sub(end_punct_pattern, "", original or "")
    corrected_core = re.sub(end_punct_pattern, "", corrected or "")

    sm = difflib.SequenceMatcher(None, original_core, corrected_core)
    result = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            result.append(corrected_core[j1:j2])
        elif tag in ('replace', 'insert'):
            result.append(f'<span class="highlight">{corrected_core[j1:j2]}</span>')
        # delete -> bỏ qua
    end_punct = corrected[-1] if re.search(end_punct_pattern, corrected or "") else ""
    return "".join(result) + end_punct

def _extract_text_from_choices(result: dict) -> str:
    """
    Cố gắng trích nội dung trả về từ nhiều định dạng khác nhau.
    """
    try:
        choices = result.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if isinstance(content, str):
            return content
        # Một số model trả dạng list-of-parts
        if isinstance(content, list):
            return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    except Exception:
        pass
    return ""

def _split_correction_and_explain(text: str):
    """
    Cố gắng tách câu sửa và phần giải thích.
    Kịch bản thường thấy: dòng đầu là câu sửa, các dòng dưới là giải thích.
    Nếu model trả theo định dạng khác thì fallback hợp lý.
    """
    t = (text or "").strip()
    if not t:
        return "", ""
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    # Thử bắt pattern kiểu "수정문: ..." hoặc "Corrected: ...":
    m = re.search(r"(수정|교정|Corrected)\s*[:：]\s*(.+)", t, flags=re.I)
    if m:
        corrected = m.group(2).strip()
        # phần còn lại (trừ dòng bắt được) là giải thích
        explain = re.sub(re.escape(m.group(0)), "", t, count=1).strip()
        return corrected, explain

    # Nếu không match, lấy dòng đầu là câu sửa, phần sau là giải thích
    corrected = lines[0]
    explain = "\n".join(lines[1:]) if len(lines) > 1 else ""
    # Loại bỏ prefix "수정:" nếu có
    corrected = re.sub(r"^.*?:\s*", "", corrected).strip()
    return corrected, explain

# ---------- main ----------
def check_grammar(text: str) -> dict:
    if not text or not text.strip():
        return {"ok": False, "error": "empty_text"}

    if not OPENROUTER_API_KEY:
        # Không có key -> trả ok=False nhưng không crash
        return {"ok": False, "error": "missing_OPENROUTER_API_KEY"}

    prompt = (
        "다음 한국어 문장에 문법/철자 오류가 있으면 자연스럽게 수정해 주세요.\n"
        "첫 줄에 '수정문: <수정된 문장>'만 출력하고, 이어서 간단한 이유를 한두 줄로 설명하세요.\n"
        f"문장: {text}"
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        # Khuyến nghị của OpenRouter (không bắt buộc, nhưng tốt cho QoS):
        # "HTTP-Referer": "http://your.site", 
        # "X-Title": "AI Korean Study Partner",
    }

    data_template = {
        "messages": [
            {"role": "system", "content": "당신은 한국어 교사입니다."},
            {"role": "user", "content": prompt},
        ]
    }

    # Retry nhẹ mỗi model
    for model in MODEL_CANDIDATES:
        if not model:
            continue
        data = dict(data_template)
        data["model"] = model

        for attempt in range(2):  # tối đa 2 lần/mô hình
            try:
                resp = requests.post(
                    OPENROUTER_URL, headers=headers, json=data,
                    timeout=(10, 30)  # 10s connect, 30s read
                )
                # Nếu HTTP status không 2xx -> trả lỗi cụ thể
                if resp.status_code // 100 != 2:
                    try:
                        j = resp.json()
                        err_str = json.dumps(j.get("error", j), ensure_ascii=False)
                    except Exception:
                        err_str = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    # Với 403 region… -> thử model tiếp theo
                    if resp.status_code in (401, 403, 429, 503):
                        break
                    else:
                        # lỗi khác -> cũng thử model tiếp theo
                        break

                result = resp.json()
                content = _extract_text_from_choices(result)
                if not content:
                    # nếu không có choices -> thử model tiếp theo
                    break

                corrected, explanation = _split_correction_and_explain(content)

                # Nếu model “không sửa” gì
                if _normalize(corrected) == _normalize(text):
                    return {
                        "ok": True,
                        "original": text,
                        "corrected": "",
                        "highlighted": "",
                        "explanation": "문법이나 철자 오류가 없습니다."
                    }

                highlighted = _highlight_changes(text, corrected)
                return {
                    "ok": True,
                    "original": text,
                    "corrected": corrected,
                    "highlighted": highlighted,
                    "explanation": explanation,
                    "model_used": model
                }

            except requests.Timeout:
                # thử lại 1 lần, rồi chuyển model
                if attempt == 0:
                    time.sleep(0.5)
                    continue
                break
            except Exception as e:
                # Lỗi mạng/parse… chuyển model tiếp
                break

    # Nếu đến đây tức tất cả model đều fail
    return {"ok": False, "error": "all_models_failed_or_region_blocked"}
