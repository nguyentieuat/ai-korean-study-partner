# grammar_checker/grammar_checker.py
import os, json, time, requests, difflib, re
from dotenv import load_dotenv

load_dotenv()

# ===== Local LLM-Service (ưu tiên) =====
GRAMMAR_LOCAL_ENABLED = os.getenv("GRAMMAR_LOCAL_ENABLED", "true").strip().lower() != "false"
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://localhost:5006").rstrip("/")
LOCAL_TIMEOUT_CONN = int(os.getenv("LOCAL_TIMEOUT_CONN", "5"))
LOCAL_TIMEOUT_READ = int(os.getenv("LOCAL_TIMEOUT_READ", "320"))

# ===== OpenRouter fallback =====
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"

MODEL_CANDIDATES = [
    os.getenv("GRAMMAR_MODEL", "").strip() or "gpt-4o-mini",
    "anthropic/claude-3.5-sonnet",
    "meta-llama/llama-3-8b-instruct",
    "meta-llama/llama-3-70b-instruct",
    "google/gemma-2-9b-it",
    "mistralai/mixtral-8x7b-instruct",
    "openai/gpt-3.5-turbo",
]

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
    end_punct = corrected[-1] if re.search(end_punct_pattern, corrected or "") else ""
    return "".join(result) + end_punct

def _split_correction_and_explain(text: str):
    t = (text or "").strip()
    if not t:
        return "", ""
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    m = re.search(r"(수정|교정|Corrected)\s*[:：]\s*(.+)", t, flags=re.I)
    if m:
        corrected = m.group(2).strip()
        explain = re.sub(re.escape(m.group(0)), "", t, count=1).strip()
        return corrected, explain
    corrected = lines[0]
    explain = "\n".join(lines[1:]) if len(lines) > 1 else ""
    corrected = re.sub(r"^.*?:\s*", "", corrected).strip()
    return corrected, explain

# ---------- LOCAL CALL (ưu tiên) ----------
def _call_local_grammar(text: str) -> dict:
    """
    Gọi llm-service local /grammar
    Body: { text: "..." }
    Trả dict kết quả giống check_grammar(); {} nếu lỗi để fallback.
    """
    if not GRAMMAR_LOCAL_ENABLED:
        return {}
    url = f"{LLM_SERVICE_URL}/grammar"
    payload = {"text": text}
    try:
        r = requests.post(url, json=payload, timeout=(LOCAL_TIMEOUT_CONN, LOCAL_TIMEOUT_READ))
        if r.status_code // 100 != 2:
            return {}
        data = r.json()
        # data đã có: {ok, original, corrected, highlighted, explanation, model_used}
        if isinstance(data, dict) and data.get("ok"):
            return data
        return {}
    except requests.RequestException:
        return {}
    except Exception:
        return {}

# ---------- OPENROUTER FALLBACK ----------
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

def _call_openrouter_grammar(text: str) -> dict:
    if not OPENROUTER_API_KEY:
        return {}

    prompt = (
        "다음 한국어 문장에 문법/철자 오류가 있으면 자연스럽게 수정해 주세요.\n"
        "첫 줄에 '수정문: <수정된 문장>'만 출력하고, 이어서 간단한 이유를 한두 줄로 설명하세요.\n"
        f"문장: {text}"
    )
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    data_template = {
        "messages": [
            {"role": "system", "content": "당신은 한국어 교사입니다."},
            {"role": "user", "content": prompt},
        ]
    }

    for model in MODEL_CANDIDATES:
        if not model:
            continue
        data = dict(data_template)
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
                    else:
                        break

                result = resp.json()
                content = _extract_text_from_choices(result)
                if not content:
                    break

                corrected, explanation = _split_correction_and_explain(content)
                if _normalize(corrected) == _normalize(text):
                    return {
                        "ok": True,
                        "original": text,
                        "corrected": "",
                        "highlighted": "",
                        "explanation": "문법이나 철자 오류가 없습니다.",
                        "model_used": model
                    }

                return {
                    "ok": True,
                    "original": text,
                    "corrected": corrected,
                    "highlighted": _highlight_changes(text, corrected),
                    "explanation": explanation,
                    "model_used": model
                }

            except requests.Timeout:
                if attempt == 0:
                    time.sleep(0.5)
                    continue
                break
            except Exception:
                break

    return {}

# ---------- PUBLIC ----------
def check_grammar(text: str) -> dict:
    if not text or not text.strip():
        return {"ok": False, "error": "empty_text"}

    # 1) thử local trước
    local_res = _call_local_grammar(text)
    if local_res:
        return local_res

    # 2) fallback OpenRouter
    or_res = _call_openrouter_grammar(text)
    if or_res:
        return or_res

    # 3) hết đường
    if GRAMMAR_LOCAL_ENABLED:
        return {"ok": False, "error": "local_and_openrouter_unavailable"}
    elif not OPENROUTER_API_KEY:
        return {"ok": False, "error": "missing_OPENROUTER_API_KEY"}
    else:
        return {"ok": False, "error": "all_models_failed_or_region_blocked"}
