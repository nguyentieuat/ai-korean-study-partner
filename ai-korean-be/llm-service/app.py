#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, requests, logging
from typing import List, Optional, Literal, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

# --- vision adapter (cache + mô tả ảnh) ---
from vision_adapter import vision_to_text_cached, VISION_MODEL

class SafeModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

load_dotenv()

# -------- config ----------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen2.5:7b-instruct")
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "600"))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("llm-service")


# -------- utils ----------
def ollama_chat(model: str, messages: List[Dict[str, str]], temperature=0.2) -> str:
    """
    Gọi Ollama /api/chat, trả content string.
    """
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "options": {"temperature": temperature},
        "stream": False,
    }
    try:
        log.debug(f"OLLAMA_CHAT: model={model}, temp={temperature}")
        r = requests.post(url, json=payload, timeout=HTTP_TIMEOUT)
        if r.status_code // 100 != 2:
            raise HTTPException(r.status_code, r.text[:1000])
        data = r.json()
        # Ollama chuẩn: {"message": {"role":"assistant","content":"..."}}
        msg = (data or {}).get("message", {})
        content = msg.get("content") or ""
        if not content and isinstance(data, dict):
            # một số backend có thể trả dạng "messages": [...]
            content = "".join([c.get("message", {}).get("content", "")
                               for c in data.get("messages", [])])
        return (content or "").strip()
    except requests.Timeout:
        raise HTTPException(504, "Ollama timeout")
    except requests.RequestException as e:
        raise HTTPException(502, f"Ollama network error: {e}")


# -------- schemas ----------
class ChatMsg(SafeModel):
    role: Literal["system","user","assistant"]
    content: str

class ChatReq(SafeModel):
    messages: List[ChatMsg]
    model: Optional[str] = None
    temperature: Optional[float] = 0.2

class ChatResp(SafeModel):
    content: str
    model_used: str


class GrammarReq(SafeModel):
    text: str
    model: Optional[str] = None

class GrammarResp(SafeModel):
    ok: bool
    original: str
    corrected: str
    highlighted: str
    explanation: str
    model_used: str


class ExplainReq(SafeModel):
    level: str
    category: str
    cau: int
    type: Optional[str] = None
    section: Optional[str] = None
    title: Optional[str] = None
    question: Optional[str] = None
    dialogue: Optional[List[Dict[str, str]]] = None
    passage: Optional[object] = None
    options: Dict[str, str]
    answer: str
    user_answer: Optional[str] = None
    language: Optional[str] = "vi"   # vi | ko | en
    model: Optional[str] = None

class ExplainResp(SafeModel):
    explanation: str
    points: List[str] = []
    language: str
    model_used: str


class VisionReq(SafeModel):
    images: List[str]
    instruction: Optional[str] = ""
    lang: Optional[str] = "ko"
    model: Optional[str] = None

class VisionResp(SafeModel):
    texts: Dict[str, str]     # mapping input -> mô tả
    model_used: str
    cached: bool = False      # true nếu tất cả ảnh lấy từ cache


# -------- app ----------
app = FastAPI(title="LLM Adapter Service (Ollama)")

@app.get("/health")
def health():
    # thử gọi /api/tags để chắc chắn Ollama online
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        ok = r.status_code // 100 == 2
    except Exception:
        ok = False
    return {"ok": ok, "ollama": OLLAMA_HOST, "model_default": DEFAULT_MODEL}

@app.get("/health/deep")
def health_deep():
    base = {"ok": False, "ollama": OLLAMA_HOST, "model_default": DEFAULT_MODEL, "chat_ok": False}
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        base["ok"] = (r.status_code // 100 == 2)
    except Exception:
        base["ok"] = False
    if base["ok"]:
        try:
            ping = ollama_chat(DEFAULT_MODEL, [{"role":"user","content":"ping"}], temperature=0.0)
            base["chat_ok"] = bool(ping)
        except Exception:
            base["chat_ok"] = False
    return base


@app.post("/chat", response_model=ChatResp)
def chat(req: ChatReq):
    model = req.model or DEFAULT_MODEL
    content = ollama_chat(model, [m.model_dump() for m in req.messages],
                          temperature=req.temperature or 0.2)
    return ChatResp(content=content, model_used=model)


# ---- Grammar: sửa ngữ pháp ngắn gọn ----
_SYSTEM_GRAMMAR = "You are a Korean language teacher."

def _grammar_prompt(text: str) -> List[Dict[str,str]]:
    user = (
        "다음 한국어 문장에 문법/철자 오류가 있으면 자연스럽게 수정하세요.\n"
        "반드시 첫 줄에 '수정문: <수정된 문장>'만 출력하고,\n"
        "그 다음 줄에 한두 문장으로 간단히 이유를 설명하세요.\n"
        f"문장: {text}"
    )
    return [
        {"role": "system", "content": _SYSTEM_GRAMMAR},
        {"role": "user", "content": user},
    ]

def _normalize(t: str) -> str:
    return re.sub(r'[.!?…]+$', '', (t or "").strip())

def _highlight(original: str, corrected: str) -> str:
    import difflib, re as _re
    end_p = r"[.?!…]$"
    o_core = re.sub(end_p, "", original or "")
    c_core = re.sub(end_p, "", corrected or "")
    sm = difflib.SequenceMatcher(None, o_core, c_core)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out.append(c_core[j1:j2])
        elif tag in ("replace","insert"):
            out.append(f'<span class="highlight">{c_core[j1:j2]}</span>')
    end = corrected[-1] if re.search(end_p, corrected or "") else ""
    return "".join(out) + end

@app.post("/grammar", response_model=GrammarResp)
def grammar(req: GrammarReq):
    model = req.model or DEFAULT_MODEL
    msgs = _grammar_prompt(req.text)
    out = ollama_chat(model, msgs, temperature=0.2)

    # tách “수정문: ...”
    corrected = ""
    explanation = ""
    m = re.search(r"(수정문)\s*[:：]\s*(.+)", out)
    if m:
        corrected = m.group(2).strip()
        explanation = re.sub(re.escape(m.group(0)), "", out, count=1).strip()
    else:
        # fallback: lấy dòng đầu là sửa
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        corrected = lines[0] if lines else ""
        explanation = "\n".join(lines[1:]) if len(lines) > 1 else ""

    # nếu chẳng khác gì đầu vào → coi như không có lỗi
    if _normalize(corrected) == _normalize(req.text):
        return GrammarResp(
            ok=True, original=req.text, corrected="",
            highlighted="", explanation="문법이나 철자 오류가 없습니다.",
            model_used=model
        )

    return GrammarResp(
        ok=True, original=req.text, corrected=corrected,
        highlighted=_highlight(req.text, corrected),
        explanation=explanation, model_used=model
    )


# ---- Explain TOPIK: build prompt DUY NHẤT ở đây ----
def _lang_name(code: str) -> str:
    return {"vi":"Tiếng Việt","en":"English","ko":"한국어"}.get(code.lower(),"Tiếng Việt")

def _estimate_limits(req: "ExplainReq") -> (int, int):
    """Ước lượng 'độ khó' để đặt số câu tối đa cho Explanation và số gạch đầu dòng."""
    passage_text = "\n".join([str(x) for x in req.passage]) if isinstance(req.passage, list) else str(req.passage or "").strip()
    dialog_text  = "\n".join([f"{d.get('speaker','')}: {d.get('text','')}" for d in (req.dialogue or [])])
    opts_text    = "\n".join([f"{k}. {v}" for k, v in sorted((req.options or {}).items())])
    q_text       = req.question or ""

    total_len = len(passage_text) + len(dialog_text) + len(opts_text) + len(q_text)

    # Heuristic đơn giản theo độ dài
    #   <= 500 ký tự: dễ  → 1–3 câu, 2 điểm
    #   501–1200:      TB → 1–5 câu, 3 điểm
    #   > 1200:        khó → 1–7 câu, 4 điểm
    if total_len > 1200:
        return 7, 4
    if total_len > 500:
        return 5, 3
    return 3, 2

def build_explain_prompt(req: "ExplainReq") -> str:
    """Xây prompt duy nhất cho explain (single content) với giới hạn câu động."""
    lang = (req.language or "vi").lower()
    lang_name = _lang_name(lang)

    passage_text = "\n".join([str(x) for x in req.passage]) if isinstance(req.passage, list) else str(req.passage or "").strip()
    dialog_text  = "\n".join([f"{d.get('speaker','')}: {d.get('text','')}" for d in (req.dialogue or [])])
    opts = "\n".join([f"{k}. {v}" for k, v in sorted((req.options or {}).items())])

    max_sents, max_points = _estimate_limits(req)

    parts = [
        f"Language: {lang_name}. WRITE ALL output in {lang_name}.",
        f"Loại: {req.type or ''} | Mục: {req.section or ''} | Level: {req.level} | Câu: {req.cau}",
    ]
    if req.title: parts.append(f"Tiêu đề: {req.title}")
    if passage_text: parts.append(f"[Đoạn văn]\n{passage_text}")
    if dialog_text:  parts.append(f"[Hội thoại]\n{dialog_text}")
    if req.question: parts.append(f"[Câu hỏi]\n{req.question}")
    parts.append(f"[Phương án]\n{opts}")
    parts.append(f"Đáp án đúng: {req.answer}")
    if req.user_answer: parts.append(f"Người dùng chọn: {req.user_answer}")

    parts.append(
        "Hãy trả về ở đúng dạng sau (đừng thêm tiền tố khác):\n"
        f"Explanation: <tối đa {max_sents} câu, ngắn gọn, mạch lạc>\n"
        "Points:\n" +
        "\n".join([f" - <điểm {i+1}>" for i in range(max_points)])
    )
    return "\n\n".join([p for p in parts if p])

_SYSTEM_EXPLAIN = (
    "You are a concise TOPIK assistant. Provide a clear, well-structured explanation and a few bullet points. "
    "Do not reveal chain-of-thought; just state conclusions and key reasons."
)

class PromptResp(SafeModel):
    prompt: str
    language: str

@app.post("/explain/prompt", response_model=PromptResp)
def explain_prompt(req: ExplainReq):
    return PromptResp(prompt=build_explain_prompt(req), language=(req.language or "vi").lower())

@app.post("/explain", response_model=ExplainResp)
def explain(req: ExplainReq):
    lang = (req.language or "vi").lower()
    model = req.model or DEFAULT_MODEL

    print(req)

    user_content = build_explain_prompt(req)
    content = ollama_chat(model, [
        {"role": "system", "content": _SYSTEM_EXPLAIN},
        {"role": "user", "content": user_content}
    ], temperature=0.2)

    # parse đơn giản theo định dạng “Explanation:” và “Points:”
    explanation = content.strip()
    points: List[str] = []
    m = re.search(r"Explanation\s*:\s*(.+?)(?:\n+Points\s*:|\Z)", content, flags=re.S)
    if m: explanation = m.group(1).strip()

    m2 = re.search(r"Points\s*:\s*(.+)\Z", content, flags=re.S)
    if m2:
        raw = m2.group(1)
        pts = re.findall(r"^[\-\*\u2022]\s*(.+)$", raw, flags=re.M)
        points = [p.strip() for p in pts if p.strip()][:3]

    return ExplainResp(explanation=explanation, points=points, language=lang, model_used=model)


# ---- Vision proxy/cache ----
@app.post("/vision2text", response_model=VisionResp)
def vision2text(req: VisionReq):
    log.info(f"VISION_REQ: {len(req.images)} images, lang={req.lang}, model={req.model}")
    texts = vision_to_text_cached(req.images, req.instruction or "", req.lang or "ko", req.model)
    cached_flag = all(bool(v) for v in texts.values())
    return {"texts": texts, "model_used": req.model or VISION_MODEL, "cached": cached_flag}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5006, reload=True)
