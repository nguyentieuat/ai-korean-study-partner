# generate_question/explanation.py
# Service: giải thích đáp án + cache sidecar (KHÔNG có FastAPI router)

import os, re, json, hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()  # tự động nạp .env trong CWD

from pydantic import BaseModel

# ==================== CONFIG (ENV) ====================
# Ghi chú: orchestration sẽ thử OpenAI trước rồi đến OpenRouter (có thể đổi thứ tự nếu muốn)
EXPLAIN_BACKEND     = os.getenv("EXPLAIN_BACKEND", "openai").lower()  # openai | openrouter
EXPLAIN_MODEL       = os.getenv("EXPLAIN_MODEL", "gpt-4o-mini")       # dùng khi backend=openai
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")

OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL    = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")
OPENROUTER_BASE     = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_SITE     = os.getenv("OPENROUTER_SITE_URL", "")
OPENROUTER_APP      = os.getenv("OPENROUTER_APP_NAME", "TOPIK Explainer")

# ==================== SCHEMAS (reusable) ====================
class ExplainReq(BaseModel):
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
    language: Optional[str] = "vi"   # "vi" | "ko" | "en"

    # write-back
    source_jsonl: Optional[str] = None
    use_sidecar: Optional[bool] = True

class ExplainResp(BaseModel):
    qid: str
    is_correct: Optional[bool]
    explanation: str
    updated: bool
    updated_path: Optional[str] = None
    explain_model: Optional[str] = None  # model actually used

# ==================== HELPERS ====================
def qid_for_question(payload: Dict[str, object]) -> str:
    """Sinh QID ổn định từ các trường ra đề cốt lõi."""
    keys = ["type", "section", "level", "title", "question", "dialogue", "passage", "options", "answer"]
    stable = {k: payload.get(k) for k in keys}
    raw = json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

def format_prompt_for_llm(q: ExplainReq) -> str:
    """Prompt ngắn gọn, yêu cầu trả JSON (không chain-of-thought)."""
    lang = (q.language or "vi").lower()
    lang_name = {"vi": "Tiếng Việt", "en": "English", "ko": "한국어"}.get(lang, "Tiếng Việt")

    head = {
        "vi": "Bạn là trợ giảng TOPIK. Hãy giải thích NGẮN GỌN và CHÍNH XÁC vì sao đáp án đúng.",
        "ko": "당신은 TOPIK 조교입니다. 정답인 이유를 짧고 정확하게 설명하세요.",
        "en": "You are a TOPIK teaching assistant. Provide a brief, precise explanation for the correct answer.",
    }.get(lang, "Bạn là trợ giảng TOPIK. Giải thích ngắn gọn và chính xác.")

    passage_text = "\n".join([str(x) for x in q.passage]) if isinstance(q.passage, list) else str(q.passage or "").strip()
    dialog_text  = "\n".join([f"{d.get('speaker','')}: {d.get('text','')}" for d in (q.dialogue or [])])
    opts = "\n".join([f"{k}. {v}" for k, v in sorted((q.options or {}).items())])

    instr = {
        "vi": "Trả về JSON: explanation (1–3 câu), rationale_points (1–3 gạch đầu dòng), language (vi/ko/en). "
              f"Ngôn ngữ trả lời: {lang_name}. VIẾT TOÀN BỘ explanation và rationale_points bằng {lang_name}.",
        "ko": f"JSON으로만 응답: explanation(1–3문장), rationale_points(1–3개), language(vi/ko/en). "
              f"응답 언어: {lang_name}. explanation과 rationale_points는 반드시 {lang_name}로 작성하세요.",
        "en": f"Return JSON only: explanation (1–3 sentences), rationale_points (1–3 bullets), language (vi/ko/en). "
              f"Response language: {lang_name}. Write ALL of explanation and rationale_points in {lang_name}.",
    }.get(lang)

    parts = [
        head,
        f"Loại: {q.type or ''} | Mục: {q.section or ''} | Level: {q.level} | Câu: {q.cau}",
    ]
    if q.title: parts.append(f"Tiêu đề: {q.title}")
    if passage_text: parts.append(f"[Đoạn văn]\n{passage_text}")
    if dialog_text:  parts.append(f"[Hội thoại]\n{dialog_text}")
    if q.question:   parts.append(f"[Câu hỏi]\n{q.question}")
    parts.append(f"[Phương án]\n{opts}")
    parts.append(f"Đáp án đúng: {q.answer}")
    if q.user_answer: parts.append(f"Người dùng chọn: {q.user_answer}")
    parts.append(instr)

    return "\n\n".join([p for p in parts if p])

def atomic_write(p: Path, data: str):
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, p)

def update_jsonl_inplace(jsonl_path: Path, qid: str, updates: Dict[str, object]) -> bool:
    """Ghi cập nhật vào file JSONL (atomically) theo qid."""
    if not jsonl_path.exists():
        return False
    changed = False
    out_lines = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            t = line.rstrip("\n")
            if not t.strip():
                out_lines.append(t); continue
            try:
                obj = json.loads(t)
            except Exception:
                out_lines.append(t); continue
            obj_qid = obj.get("qid") or qid_for_question(obj)
            if obj_qid == qid:
                obj["qid"] = qid
                obj.update(updates)
                changed = True
                out_lines.append(json.dumps(obj, ensure_ascii=False))
            else:
                out_lines.append(json.dumps(obj, ensure_ascii=False))
    if changed:
        atomic_write(jsonl_path, "\n".join(out_lines) + "\n")
    return changed

# -------------------- Sidecar (đa ngôn ngữ) --------------------
def _sidecar_path(qid: str) -> Path:
    side_dir = Path("data/explanations")
    side_dir.mkdir(parents=True, exist_ok=True)
    return side_dir / f"{qid}.json"

def _migrate_old_format_if_needed(p: Path, data: Dict[str, object]) -> Dict[str, object]:
    """
    Nếu file còn field cũ 'explanation_ai' thì chuyển sang:
      { "explanations": { <lang>: {text, points, ...} } }
    """
    if "explanations" in data:
        return data
    old = data.get("explanation_ai")
    if not isinstance(old, dict):
        return data
    lang = (old.get("lang") or "vi").lower()
    exps = {lang: old}
    new_data = {"qid": data.get("qid"), "explanations": exps}
    # merge mọi key khác (trừ explanation_ai)
    for k, v in data.items():
        if k not in ("explanation_ai", "explanations"):
            new_data[k] = v
    atomic_write(p, json.dumps(new_data, ensure_ascii=False, indent=2))
    return new_data

def load_sidecar_explanation(qid: str, lang: str) -> Optional[Dict[str, object]]:
    """
    Trả về block giải thích cho ngôn ngữ yêu cầu (nếu có).
    Ưu tiên 'explanations[lang]'. Có migrate định dạng cũ nếu cần.
    """
    p = _sidecar_path(qid)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        data = _migrate_old_format_if_needed(p, data)
        exps = data.get("explanations") or {}
        return exps.get((lang or "vi").lower())
    except Exception:
        return None

def write_sidecar_lang(qid: str, lang: str, entry: Dict[str, object]) -> Path:
    """
    Ghi/ghi-đè CHỈ ngôn ngữ 'lang' vào sidecar, không đụng các ngôn ngữ khác.
    entry: {text, points, lang, model, backend, ts}
    """
    p = _sidecar_path(qid)
    content: Dict[str, object] = {"qid": qid, "explanations": {}}
    if p.exists():
        try:
            content = json.loads(p.read_text(encoding="utf-8"))
            content = _migrate_old_format_if_needed(p, content)
            if "explanations" not in content or not isinstance(content["explanations"], dict):
                content["explanations"] = {}
        except Exception:
            content = {"qid": qid, "explanations": {}}
    content["qid"] = qid
    content["explanations"][ (lang or "vi").lower() ] = entry
    atomic_write(p, json.dumps(content, ensure_ascii=False, indent=2))
    return p

# ---- Validate text hợp lệ để dùng cache / chấp nhận backend kết quả ----
def _is_valid_explain_text(txt: str) -> bool:
    if not txt:
        return False
    bad = [
        "missing",
        "không có openai_api_key",
        "openai_api_key missing",
        "chưa có openrouter_api_key",
        "no automatic explanation",
        "hiện chưa có lời giải thích",
        "lỗi gọi",
        "error",
    ]
    t = txt.lower()
    return not any(b in t for b in bad)

# ==================== BACKENDS ====================
def call_llm_openai(prompt: str, language: str) -> Dict[str, object]:
    if not OPENAI_API_KEY:
        return {"explanation": {
            "vi": "Không có OPENAI_API_KEY.",
            "en": "OPENAI_API_KEY missing."
        }.get((language or "vi").lower())}
    try:
        from openai import OpenAI
    except Exception as e:
        return {"explanation": f"Lỗi gọi OpenAI: {e}"}
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        sysmsg = (
            "You are a concise TOPIK assistant. Return ONLY a valid JSON object "
            "with keys: explanation (1–3 sentences), rationale_points (1–3 items), language (vi/ko/en)."
        )
        resp = client.chat.completions.create(
            model=EXPLAIN_MODEL,
            messages=[
                {"role": "system", "content": sysmsg},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        txt = resp.choices[0].message.content
        data = json.loads(txt)
        return data if isinstance(data, dict) else {"explanation": str(data)}
    except Exception as e:
        # ví dụ quota 429, network error…
        return {"explanation": f"Lỗi gọi OpenAI: {e}"}

def call_llm_openrouter(prompt: str, language: str) -> Dict[str, object]:
    if not OPENROUTER_API_KEY:
        return {"explanation": {
            "vi":"Chưa có OPENROUTER_API_KEY.",
            "en":"OPENROUTER_API_KEY missing."
        }.get((language or "vi").lower())}
    import requests
    sysmsg = (
        "You are a concise TOPIK assistant. Return ONLY a valid JSON object "
        "with keys: explanation (1–3 sentences), rationale_points (1–3 items), language (vi/ko/en)."
    )
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    if OPENROUTER_SITE: headers["HTTP-Referer"] = OPENROUTER_SITE
    if OPENROUTER_APP:  headers["X-Title"] = OPENROUTER_APP
    url = f"{OPENROUTER_BASE.rstrip('/')}/chat/completions"
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role":"system","content":sysmsg},
            {"role":"user","content":prompt}
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    def _parse(txt: str) -> Dict[str, object]:
        try:
            obj = json.loads(txt)
            return obj if isinstance(obj, dict) and "explanation" in obj else {"explanation": str(obj)}
        except Exception:
            m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
            return json.loads(m.group(0)) if m else {"explanation": txt[:500]}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"]
        return _parse(txt)
    except Exception:
        # nới lỏng nếu model từ chối response_format
        payload.pop("response_format", None)
        payload["messages"][0]["content"] = sysmsg + " Return only JSON."
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"]
            return _parse(txt)
        except Exception as e2:
            return {"explanation": f"Lỗi gọi OpenRouter: {e2}"}

# ---- Orchestrator: ưu tiên OpenAI -> OpenRouter, lấy backend dùng thực tế ----
def call_llm_explainer(prompt: str, language: str) -> Tuple[Dict[str, object], str, str]:
    """
    Trả về (out_dict, backend_used, model_used).
    Thứ tự ưu tiên: OpenAI trước, nếu fail/placeholder thì thử OpenRouter.
    """
    # order = ["openai", "openrouter"]
    order = ["openrouter", "openai"]
    last_err = None

    for backend in order:
        try:
            if backend == "openai":
                out = call_llm_openai(prompt, language)
                model_used = EXPLAIN_MODEL
            else:
                out = call_llm_openrouter(prompt, language)
                model_used = OPENROUTER_MODEL

            if isinstance(out, dict):
                text = str(out.get("explanation", "")).strip()
                if _is_valid_explain_text(text):
                    return out, backend, model_used
                else:
                    last_err = f"{backend} returned invalid/placeholder text"
            else:
                last_err = f"{backend} returned non-dict result"
        except Exception as e:
            last_err = f"{backend} exception: {e}"

    # Không backend nào ok
    fallback = {
        "vi": f"Không thể sinh lời giải tự động ({last_err}).",
        "en": f"Unable to generate explanation ({last_err}).",
        "ko": f"자동 해설을 생성할 수 없습니다 ({last_err})."
    }
    lang = (language or "vi").lower()
    return {"explanation": fallback.get(lang, fallback["vi"])}, "none", "none"

# ==================== PUBLIC SERVICE API ====================
def explain_question(body: ExplainReq, mode: str = "auto") -> ExplainResp:
    """
    Service chính: trả ExplainResp (không phụ thuộc FastAPI).
    mode: auto | cache | force
    """
    # 1) QID
    base_payload = {
        "type": body.type, "section": body.section, "level": body.level,
        "title": body.title, "question": body.question,
        "dialogue": body.dialogue, "passage": body.passage,
        "options": body.options, "answer": body.answer,
    }
    qid = qid_for_question(base_payload)
    req_lang = (body.language or "vi").lower()

    # 2) Cache-first (theo ngôn ngữ)
    side = load_sidecar_explanation(qid, req_lang)
    if mode in ("auto", "cache") and side and _is_valid_explain_text(side.get("text","")):
        return ExplainResp(
            qid=qid,
            is_correct=(None if not body.user_answer else str(body.user_answer).strip().upper()==str(body.answer).strip().upper()),
            explanation=side.get("text") or "",
            updated=False,
            updated_path=str(_sidecar_path(qid)),
            explain_model=side.get("model"),
        )
    if mode == "cache":
        return ExplainResp(
            qid=qid, is_correct=None, explanation="(Chưa có giải thích trong cache.)",
            updated=False, updated_path=None, explain_model=None
        )

    # 3) Gọi LLM (OpenAI -> OpenRouter)
    prompt = format_prompt_for_llm(body)
    llm_out, backend_used, model_used = call_llm_explainer(prompt, req_lang)

    # Chuẩn hoá
    explanation = ""
    rationale_points = None
    if isinstance(llm_out, dict):
        explanation = str(llm_out.get("explanation", "")).strip()
        rp = llm_out.get("rationale_points")
        if isinstance(rp, list):
            rationale_points = [str(x) for x in rp][:3]
    if not explanation:
        explanation = {
            "vi": "Hiện chưa có lời giải thích tự động.",
            "en": "No automatic explanation.",
            "ko": "자동 해설이 없습니다.",
        }.get(req_lang, "Hiện chưa có lời giải thích tự động.")

    # 4) Write-back (ghi theo ngôn ngữ)
    now_iso = datetime.utcnow().isoformat() + "Z"
    entry = {
        "text": explanation,
        "points": rationale_points or [],
        "lang": req_lang,
        "model": model_used,
        "backend": backend_used,
        "ts": now_iso,
    }

    updated = False
    updated_path = None
    if body.use_sidecar or not body.source_jsonl:
        p = write_sidecar_lang(qid, req_lang, entry)
        updated, updated_path = True, str(p)
    else:
        # JSONL thì vẫn merge theo qid; (tuỳ bạn muốn giữ đa ngôn ngữ thế nào trong JSONL)
        updates = {"qid": qid, "explanations": {req_lang: entry}}
        p = Path(body.source_jsonl)
        updated = update_jsonl_inplace(p, qid, updates)
        updated_path = str(p) if updated else None

    return ExplainResp(
        qid=qid,
        is_correct=(None if not body.user_answer else str(body.user_answer).strip().upper()==str(body.answer).strip().upper()),
        explanation=explanation,
        updated=updated,
        updated_path=updated_path,
        explain_model=model_used,
    )
