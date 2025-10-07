#generate_question/explanation.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Service: giải thích đáp án + cache sidecar (KHÔNG có FastAPI router)

import os, re, json, hashlib, logging, base64
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

import requests  # dùng cho local llm-service & openrouter

load_dotenv()  # tự động nạp .env trong CWD

from pydantic import BaseModel

# ==================== LOGGING ====================
LOG_LEVEL = os.getenv("EXPLAIN_LOG_LEVEL", "INFO").upper()
_logger_initialized = getattr(logging, "_explain_inited", False)
if not _logger_initialized:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging._explain_inited = True
log = logging.getLogger("topik.explain")

def _redact(seen: object) -> object:
    try:
        if isinstance(seen, dict):
            red = {}
            for k, v in seen.items():
                if any(token in k.lower() for token in ["api_key", "authorization", "token"]):
                    red[k] = "***REDACTED***"
                else:
                    red[k] = _redact(v)
            return red
        if isinstance(seen, list):
            return [_redact(x) for x in seen]
    except Exception:
        pass
    return seen

# ==================== CONFIG (ENV) ====================
# Ưu tiên LOCAL -> OpenRouter (không gọi OpenAI)
LLM_LOCAL_URL   = os.getenv("LLM_LOCAL_URL", "http://llm_service:5006/explain").rstrip("/")

OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL    = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")
OPENROUTER_BASE     = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_SITE     = os.getenv("OPENROUTER_SITE_URL", "")
OPENROUTER_APP      = os.getenv("OPENROUTER_APP_NAME", "TOPIK Explainer")

# Vision RPC (llm-service chịu trách nhiệm cache + dịch)
VISION_RPC_URL      = os.getenv("VISION_RPC_URL", "http://llm_service:5006/vision2text").rstrip("/")
VISION_RPC_TIMEOUT  = int(os.getenv("VISION_RPC_TIMEOUT", "600"))

# Kho record mô tả theo key (per-key file)
VISION_RECORDS_DIR  = os.getenv("VISION_RECORDS_DIR", "data/vision_records")
# Ngôn ngữ mặc định của ĐỀ/desc trong kho record & khi fallback RPC (KHÔNG phải ngôn ngữ giải thích cho user)
VISION_RECORDS_LANG_DEFAULT = os.getenv("VISION_RECORDS_LANG_DEFAULT", "ko").lower()

# Bật để gắn nhãn nguồn mô tả (records-key/rpc/embedded/label-only) ngay trong chuỗi kết quả
MEDIA_DEBUG = int(os.getenv("EXPLAIN_MEDIA_DEBUG", "0"))

log.info(
    "Explain config | local_url=%s, or_model=%s, log_level=%s, vision_rpc=%s, records_dir=%s, records_lang=%s",
    LLM_LOCAL_URL, OPENROUTER_MODEL, LOG_LEVEL, VISION_RPC_URL, VISION_RECORDS_DIR, VISION_RECORDS_LANG_DEFAULT
)

# ==================== SCHEMAS ====================
class ExplainReq(BaseModel):
    level: str
    category: str
    cau: int
    type: Optional[str] = None
    section: Optional[str] = None
    title: Optional[str] = None
    question: Optional[str] = None
    dialogue: Optional[List[Dict[str, str]]] = None  # [{"speaker":..., "text":...}] hoặc {"image": "..."}
    passage: Optional[object] = None
    options: Dict[str, str]
    answer: str
    user_answer: Optional[str] = None
    language: Optional[str] = "vi"   # "vi" | "ko" | "en" (ngôn ngữ giải thích cho người dùng)

    # write-back
    source_jsonl: Optional[str] = None
    use_sidecar: Optional[bool] = True

class ExplainResp(BaseModel):
    qid: str
    is_correct: Optional[bool]
    explanation: str
    updated: bool
    updated_path: Optional[str] = None
    explain_model: Optional[str] = None
    errors: Optional[List[Dict[str, object]]] = None

# ==================== HELPERS ====================
def qid_for_question(payload: Dict[str, object]) -> str:
    keys = ["type", "section", "level", "title", "question", "dialogue", "passage", "options", "answer"]
    stable = {k: payload.get(k) for k in keys}
    raw = json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

# -------- Vision helpers (client) --------
_IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")

def _is_imagestr(s: Optional[str]) -> bool:
    if not s or not isinstance(s, str): return False
    s2 = s.strip().lower()
    return s2.startswith("data:image/") or s2.endswith(_IMG_EXT) or s2.startswith("http://") or s2.startswith("https://")

def _file_to_b64(p: str) -> Optional[str]:
    try:
        data = Path(p).read_bytes()
        return base64.b64encode(data).decode("ascii")
    except Exception as e:
        log.warning("vision2text(file->b64) failed %s: %s", p, e)
        return None

def _image_ref_to_b64(image_ref: str) -> Optional[str]:
    """
    dataURL -> trả b64; local path -> đọc -> b64.
    URL http(s): không tải ở đây; để llm-service tự xử nếu cần (khuyến nghị gửi b64).
    """
    if not image_ref or not isinstance(image_ref, str): return None
    s = image_ref.strip()
    if s.startswith("data:image/"):
        try:
            return s.split(",", 1)[1]
        except Exception:
            return None
    if s.lower().endswith(_IMG_EXT) and Path(s).exists():
        return _file_to_b64(s)
    # Bỏ qua http(s)
    return None

def _vision2text_rpc_batch(images_b64: List[str], lang: str, instruction: str = "") -> List[str]:
    """
    RPC đến llm-service /vision2text (server lo cache/dịch).
    Request: { "images": [...], "lang": "ko", "instruction": "" }
    """
    if not images_b64: return []
    payload = {"images": images_b64, "lang": (lang or "ko").lower(), "instruction": instruction or ""}
    try:
        r = requests.post(VISION_RPC_URL, json=payload, timeout=VISION_RPC_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning("vision rpc error: %s", e)
        return [""] * len(images_b64)

    n = len(images_b64); texts: List[str] = []
    if isinstance(data, dict):
        if isinstance(data.get("texts"), dict):
            # map input->text: giữ thứ tự theo mảng vào (0..n-1)
            # nhưng vì ta gửi mảng images_b64 (không có key), server hiện trả dict (map theo input string)
            # ở đây gom theo thứ tự đã gửi:
            for i in range(n):
                key = images_b64[i]
                texts.append(str(data["texts"].get(key, "")).strip())
        elif isinstance(data.get("texts"), list):
            texts = [str(x or "").strip() for x in data["texts"]]
        elif isinstance(data.get("results"), list):
            texts = [str((x or {}).get("text", "")).strip() for x in data["results"]]
        elif isinstance(data.get("map"), dict):
            texts = [str(data["map"].get(str(i), "")).strip() for i in range(n)]
    if len(texts) < n: texts += [""] * (n - len(texts))
    return texts[:n]

def _lang_label(lang: str) -> str:
    return {"ko": "[이미지]", "vi": "[ẢNH]", "en": "[IMG]"}.get((lang or "ko").lower(), "[이미지]")

# ====== helpers ưu tiên lấy desc theo key từ data/vision_records ======
def _hex40(s: str) -> bool:
    try:
        return bool(re.fullmatch(r"[0-9a-fA-F]{40}", s or ""))
    except Exception:
        return False

def _records_path_for_key(key: str) -> Path:
    base = Path(VISION_RECORDS_DIR)
    p_flat = base / f"{key}.json"
    if p_flat.exists():
        return p_flat
    return base / key[:2] / key[2:4] / f"{key}.json"

def _load_desc_by_key(key: str, lang: str) -> Optional[str]:
    """Đọc desc theo lang từ record per-key. Ưu tiên desc[lang], sau đó thử các trường khác."""
    if not key or not _hex40(key): return None
    p = _records_path_for_key(key)
    if not p.exists(): return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("vision_records parse error key=%s file=%s err=%s", key, str(p), e)
        return None

    # 1) Desc đa ngôn ngữ
    desc = (obj or {}).get("desc")
    if isinstance(desc, dict):
        want = (lang or "ko").lower()
        txt = (desc.get(want) or "").strip()
        if txt: return txt
        # fallback vòng: ko -> vi -> en
        for fb in ["ko", "vi", "en"]:
            if fb == want: continue
            t2 = (desc.get(fb) or "").strip()
            if t2: return t2

    # 2) Trường khác nếu có
    for k in ("vision_desc", "ocr_text", "caption"):
        t = str((obj or {}).get(k) or "").strip()
        if t: return t

    return None

# ====== base64 helpers (MỚI) ======
_B64_RE = re.compile(r'^[A-Za-z0-9+/]+={0,2}$')

def _looks_like_raw_b64(s: str) -> bool:
    if not isinstance(s, str): return False
    s = s.strip()
    if len(s) < 64 or (len(s) % 4) != 0:
        return False
    if not _B64_RE.fullmatch(s):
        return False
    try:
        base64.b64decode(s, validate=True)
        return True
    except Exception:
        return False

def _extract_b64_any(x: object) -> Optional[str]:
    """Trả lại base64 (không prefix) từ nhiều dạng đầu vào: dataURL/raw-b64/file path/dict."""
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("data:image/"):
            parts = s.split(",", 1)
            if len(parts) == 2 and parts[1].strip():
                return parts[1].strip()
            return None
        if _looks_like_raw_b64(s):
            return s
        if _is_imagestr(s):  # path or http(s); http bị bỏ qua trong _image_ref_to_b64
            return _image_ref_to_b64(s)
        return None
    if isinstance(x, dict):
        for k in ("image_b64", "b64", "base64"):
            v = x.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        for k in ("image", "src", "text"):
            v = x.get(k)
            b = _extract_b64_any(v)
            if b:
                return b
    return None

def _extract_image_key(x: object) -> Optional[str]:
    """Lấy key theo thứ tự: explicit key -> tính từ base64(raw) bằng sha1(base64_string)."""
    # explicit key
    if isinstance(x, dict):
        for k in ("image_key", "key"):
            v = x.get(k)
            if isinstance(v, str) and _hex40(v):
                return v.lower()
        img = x.get("image") or x.get("src")
        if isinstance(img, dict):
            for k in ("image_key", "key"):
                v = img.get(k)
                if isinstance(v, str) and _hex40(v):
                    return v.lower()
        v = x.get("text")
        if isinstance(v, str) and _hex40(v):
            return v.lower()
    elif isinstance(x, str):
        s = x.strip()
        if _hex40(s):
            return s.lower()
        if s.lower().startswith("key:"):
            s2 = s[4:].strip()
            if _hex40(s2):
                return s2.lower()

    # tính từ base64 (điểm mấu chốt để support raw b64)
    b64 = _extract_b64_any(x)
    if b64:
        try:
            # theo yêu cầu: sha1(base64(raw_bytes))
            return hashlib.sha1(b64.encode("utf-8")).hexdigest()
        except Exception:
            return None

    return None

def _image_any_to_b64_if_needed(x: object) -> Optional[str]:
    """Trả b64 nếu suy ra được (kể cả khi có key) để có thể fallback RPC khi record thiếu."""
    return _extract_b64_any(x)

def _desc_from_obj(obj: object, lang: str) -> Optional[str]:
    """
    Lấy desc trực tiếp từ object nếu có:
      obj = {"desc": {"ko": "...", "vi": "...", "en": "..."}, ...}
    Ưu tiên lang, rồi ko -> vi -> en.
    """
    if not isinstance(obj, dict):
        return None
    desc = obj.get("desc")
    if isinstance(desc, dict):
        want = (lang or "ko").lower()
        txt = (desc.get(want) or "").strip()
        if txt:
            return txt
        for fb in ["ko", "vi", "en"]:
            if fb == want:
                continue
            t2 = (desc.get(fb) or "").strip()
            if t2:
                return t2
    return None

def _mk_rep(label: str, text: str, src: str, key: Optional[str] = None) -> str:
    rep = f"{label} {text}".strip() if (text or "").strip() else f"{label}"
    if MEDIA_DEBUG:
        tag = src if not key else f"{src}:{key[:8]}"
        rep += f" ‹{tag}›"
    return rep


# ================== PREPROCESS (UPDATED) ==================
def _preprocess_media_in_body(body: "ExplainReq") -> "ExplainReq":
    """
    Ưu tiên lấy mô tả từ data/vision_records theo key (image_key/key) bằng ngôn ngữ ĐỀ (mặc định ko).
    Nếu không có, gom ảnh còn thiếu và gọi Vision RPC (lang = ko) theo batch.
    Thay thế ảnh bằng chuỗi: "[LABEL] {desc}" với LABEL cùng ngôn ngữ desc (mặc định ko).
    """
    new = ExplainReq(**body.dict())

    # Ngôn ngữ để lấy mô tả đề & gọi RPC (KHÔNG dùng language của request)
    record_lang = (VISION_RECORDS_LANG_DEFAULT or "ko").lower()

    # LABEL hiển thị: mặc định cùng ngôn ngữ với desc (record_lang).
    # Có thể override qua env nếu cần (ví dụ muốn nhãn [ẢNH] cho vi)
    label_lang = os.getenv("VISION_LABEL_LANG_DEFAULT", record_lang).lower()
    label = _lang_label(label_lang)

    b64s: List[str] = []
    to_describe: List[Tuple[str, int, Optional[str]]] = []  # (kind, idx, opt_key)

    def _replace_with_desc(target_kind: str, idx: int, opt_key: Optional[str], source_obj: object):
        # 0) ƯU TIÊN desc NHÚNG TRONG OBJECT (nếu có)
        embedded = _desc_from_obj(source_obj, record_lang)
        if embedded:
            return _mk_rep(label, embedded, src="embedded")

        # 1) Theo key (ưu tiên tiếp theo)
        key = _extract_image_key(source_obj)
        if key:
            desc_txt = _load_desc_by_key(key, record_lang)
            if desc_txt:
                return _mk_rep(label, desc_txt, src="records-key", key=key)

        # 2) Gom cho RPC nếu cần (chỉ khi chưa có desc)
        b64 = _image_any_to_b64_if_needed(source_obj)
        if b64:
            to_describe.append((target_kind, idx, opt_key))
            b64s.append(b64)
            return None

        # 3) Không có data nào hữu ích -> trả label rỗng
        return _mk_rep(label, "", src="label-only")

    # ---- PASSAGE ----
    if new.passage:
        if isinstance(new.passage, str):
            s = new.passage.strip()
            if _is_imagestr(s) or _hex40(s) or s.lower().startswith("key:") or _looks_like_raw_b64(s):
                rep = _replace_with_desc("pass", -1, None, new.passage)
                if rep is not None:
                    new.passage = rep
        elif isinstance(new.passage, list):
            for i, it in enumerate(new.passage):
                is_img_like = False
                if isinstance(it, str) and (_is_imagestr(it) or _hex40(it.strip()) or it.lower().startswith("key:") or _looks_like_raw_b64(it)):
                    is_img_like = True
                elif isinstance(it, dict):
                    if (_extract_image_key(it)
                        or _is_imagestr(it.get("image"))
                        or _is_imagestr(it.get("text"))
                        or _is_imagestr(it.get("src"))
                        or it.get("image_b64") or it.get("b64") or it.get("base64")):
                        is_img_like = True
                        if isinstance(it.get("text"), str) and _is_imagestr(it.get("text")):
                            it["text"] = ""
                if is_img_like:
                    rep = _replace_with_desc("pass", i, None, it)
                    if rep is not None:
                        new.passage[i] = rep
        elif isinstance(new.passage, dict):
            it = new.passage
            if (_extract_image_key(it)
                or _is_imagestr(it.get("image"))
                or _is_imagestr(it.get("text"))
                or _is_imagestr(it.get("src"))
                or it.get("image_b64") or it.get("b64") or it.get("base64")):
                if isinstance(it.get("text"), str) and _is_imagestr(it.get("text")):
                    it["text"] = ""
                rep = _replace_with_desc("pass", -2, None, it)
                if rep is not None:
                    new.passage = rep

    # ---- OPTIONS ----
    if isinstance(new.options, dict):
        for k, v in list(new.options.items()):
            is_img_like = False
            if isinstance(v, str) and (_is_imagestr(v) or _hex40(v.strip()) or v.lower().startswith("key:") or _looks_like_raw_b64(v)):
                is_img_like = True
            elif isinstance(v, dict):
                if (_extract_image_key(v)
                    or _is_imagestr(v.get("image"))
                    or _is_imagestr(v.get("text"))
                    or _is_imagestr(v.get("src"))
                    or v.get("image_b64") or v.get("b64") or v.get("base64")):
                    is_img_like = True
                    if isinstance(v.get("text"), str) and _is_imagestr(v.get("text")):
                        v["text"] = ""
            if is_img_like:
                rep = _replace_with_desc("opt", -1, k, v)
                if rep is not None:
                    new.options[k] = rep

    # ---- RPC cho phần còn thiếu (lang = record_lang) ----
    if b64s:
        descs: List[str] = _vision2text_rpc_batch(b64s, lang=record_lang, instruction="")
        ptr = 0

        if isinstance(new.passage, str):
            if any(kind == "pass" and i == -1 for kind, i, _ in to_describe):
                text = (descs[ptr] if ptr < len(descs) else "").strip(); ptr += 1
                new.passage = _mk_rep(label, text, src="rpc")
        elif isinstance(new.passage, list):
            for kind, i, _ in to_describe:
                if kind == "pass" and i >= 0:
                    text = (descs[ptr] if ptr < len(descs) else "").strip(); ptr += 1
                    new.passage[i] = _mk_rep(label, text, src="rpc")
        elif isinstance(new.passage, dict):
            if any(kind == "pass" and i == -2 for kind, i, _ in to_describe):
                text = (descs[ptr] if ptr < len(descs) else "").strip(); ptr += 1
                new.passage = _mk_rep(label, text, src="rpc")

        if isinstance(new.options, dict):
            for kind, _, optkey in to_describe:
                if kind == "opt" and optkey in new.options:
                    text = (descs[ptr] if ptr < len(descs) else "").strip(); ptr += 1
                    new.options[optkey] = _mk_rep(label, text, src="rpc")

    return new

# ================== PROMPT (legacy fallback; KHÔNG dùng trong flow thường) ==================
def format_prompt_for_llm(q: ExplainReq) -> str:
    """(Safety net) Prompt builder cũ — chỉ dùng khi không lấy được prompt từ llm-service."""
    q2 = _preprocess_media_in_body(q)  # để đảm bảo media đã được thay mô tả nếu có
    lang = (q2.language or "vi").lower()
    lang_name = {"vi": "Tiếng Việt", "en": "English", "ko": "한국어"}.get(lang, "Tiếng Việt")

    head = {
        "vi": "Bạn là trợ giảng TOPIK. Hãy giải thích NGẮN GỌN và CHÍNH XÁC vì sao đáp án đúng.",
        "ko": "당신은 TOPIK 조교입니다. 정답인 이유를 짧고 정확하게 설명하세요.",
        "en": "You are a TOPIK teaching assistant. Provide a brief, precise explanation for the correct answer.",
    }.get(lang, "Bạn là trợ giảng TOPIK. Giải thích ngắn gọn và chính xác.")

    passage_text = "\n".join([str(x) for x in q2.passage]) if isinstance(q2.passage, list) else str(q2.passage or "").strip()
    dialog_text  = "\n".join([f"{d.get('speaker','')}: {d.get('text','')}" for d in (q2.dialogue or [])])
    opts = "\n".join([f"{k}. {v}" for k, v in sorted((q2.options or {}).items())])

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
        f"Loại: {q2.type or ''} | Mục: {q2.section or ''} | Level: {q2.level} | Câu: {q2.cau}"
    ]
    if q2.title: parts.append(f"Tiêu đề: {q2.title}")
    if passage_text: parts.append(f"[Đoạn văn]\n{passage_text}")
    if dialog_text:  parts.append(f"[Hội thoại]\n{dialog_text}")
    if q2.question:  parts.append(f"[Câu hỏi]\n{q2.question}")
    parts.append(f"[Phương án]\n{opts}")
    parts.append(f"Đáp án đúng: {q2.answer}")
    if q2.user_answer: parts.append(f"Người dùng chọn: {q2.user_answer}")
    parts.append(instr)

    return "\n\n".join([p for p in parts if p])


def atomic_write(p: Path, data: str):
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, p)

def update_jsonl_inplace(jsonl_path: Path, qid: str, updates: Dict[str, object]) -> bool:
    if not jsonl_path.exists():
        log.warning("JSONL not found for update: %s", str(jsonl_path))
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
                obj.update(updates); changed = True
                out_lines.append(json.dumps(obj, ensure_ascii=False))
            else:
                out_lines.append(json.dumps(obj, ensure_ascii=False))
    if changed:
        atomic_write(jsonl_path, "\n".join(out_lines) + "\n")
        log.info("JSONL updated inplace for qid=%s at %s", qid, str(jsonl_path))
    else:
        log.info("JSONL unchanged (qid not found) for %s", str(jsonl_path))
    return changed


# -------------------- Sidecar (đa ngôn ngữ) --------------------
def _sidecar_path(qid: str) -> Path:
    side_dir = Path("data/explanations"); side_dir.mkdir(parents=True, exist_ok=True)
    return side_dir / f"{qid}.json"

def _migrate_old_format_if_needed(p: Path, data: Dict[str, object]) -> Dict[str, object]:
    if "explanations" in data: return data
    old = data.get("explanation_ai")
    if not isinstance(old, dict): return data
    lang = (old.get("lang") or "vi").lower()
    new_data = {"qid": data.get("qid"), "explanations": {lang: old}}
    for k, v in data.items():
        if k not in ("explanation_ai", "explanations"):
            new_data[k] = v
    atomic_write(p, json.dumps(new_data, ensure_ascii=False, indent=2))
    log.info("Migrated sidecar to multi-lang: %s", str(p))
    return new_data

def load_sidecar_explanation(qid: str, lang: str) -> Optional[Dict[str, object]]:
    p = _sidecar_path(qid)
    if not p.exists(): return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        data = _migrate_old_format_if_needed(p, data)
        return (data.get("explanations") or {}).get((lang or "vi").lower())
    except Exception as e:
        log.error("Sidecar read error qid=%s file=%s err=%s", qid, str(p), e)
        return None

def write_sidecar_lang(qid: str, lang: str, entry: Dict[str, object]) -> Path:
    p = _sidecar_path(qid)
    content: Dict[str, object] = {"qid": qid, "explanations": {}}
    if p.exists():
        try:
            content = json.loads(p.read_text(encoding="utf-8"))
            content = _migrate_old_format_if_needed(p, content)
            if "explanations" not in content or not isinstance(content["explanations"], dict):
                content["explanations"] = {}
        except Exception as e:
            log.warning("Sidecar parse error; overwrite. file=%s err=%s", str(p), e)
            content = {"qid": qid, "explanations": {}}
    content["qid"] = qid
    content["explanations"][(lang or "vi").lower()] = entry
    atomic_write(p, json.dumps(content, ensure_ascii=False, indent=2))
    log.info("Sidecar wrote qid=%s lang=%s -> %s", qid, lang, str(p))
    return p


# ---- Validate text ----
def _is_valid_explain_text(txt: str) -> bool:
    if not txt: return False
    bad = [
        "missing","không có openrouter_api_key","openai_api_key missing","chưa có openrouter_api_key",
        "no automatic explanation","hiện chưa có lời giải thích","lỗi gọi","error","quota","429",
        "insufficient_quota","unable to generate explanation","không thể sinh lời tự động",
        "không thể sinh lời giải tự động",
    ]
    t = txt.lower()
    return not any(b in t for b in bad)


# ==================== BACKENDS ====================
def _err(backend: str, model: str, kind: str, message: str,
         status: Optional[int] = None, code: Optional[str] = None,
         raw: Optional[object] = None) -> Dict[str, object]:
    obj = {"backend": backend,"model": model,"kind": kind,"status": status,"code": code,
           "message": message,"raw": raw if isinstance(raw,(str,int,float,type(None))) else _redact(raw)}
    lvl = logging.ERROR if kind in ("http_error","sdk_error","network_error") else logging.WARNING
    log.log(lvl, "LLM error | %s", obj)
    return obj

# ---- Local LLM-Service (ưu tiên) ----
def _local_base_url() -> str:
    # LLM_LOCAL_URL mặc định là ".../explain" → thay bằng gốc để gọi /explain & /explain/prompt
    if LLM_LOCAL_URL.endswith("/explain"):
        return LLM_LOCAL_URL[:-len("/explain")]
    return LLM_LOCAL_URL.rsplit("/", 1)[0]

def call_llm_local(body: ExplainReq) -> Dict[str, object]:
    try:
        url = _local_base_url().rstrip("/") + "/explain"
        r = requests.post(url, json=body.dict(), timeout=360)
        if r.status_code >= 400:
            return {"error": _err("local", "llm-service", "http_error", "HTTP error",
                                  status=r.status_code, raw=r.text[:500])}
        data = r.json()
        explanation = data.get("explanation") or ""
        if not explanation:
            return {"error": _err("local","llm-service","invalid_output","Missing explanation", raw=data)}
        return {
            "explanation": explanation,
            "rationale_points": data.get("points", []),
            "language": data.get("language", body.language),
            "model": data.get("model_used", "local"),
        }
    except requests.RequestException as e:
        return {"error": _err("local", "llm-service", "network_error", str(e))}

def fetch_prompt_from_local(body: ExplainReq) -> Optional[str]:
    """Xin prompt chuẩn từ llm-service để tránh drift định dạng."""
    try:
        prompt_url = _local_base_url().rstrip("/") + "/explain/prompt"
        r = requests.post(prompt_url, json=body.dict(), timeout=360)
        if r.status_code // 100 == 2:
            data = r.json()
            return (data or {}).get("prompt")
    except Exception as e:
        log.warning("fetch_prompt_from_local failed: %s", e)
    return None

# ---- OpenRouter (fallback) ----
def call_llm_openrouter(prompt: str, language: str) -> Dict[str, object]:
    if not OPENROUTER_API_KEY:
        return {"error": _err("openrouter", OPENROUTER_MODEL, "no_key", "OPENROUTER_API_KEY missing")}
    sysmsg = ("You are a concise TOPIK assistant. Return ONLY a valid JSON object "
              "with keys: explanation (1–3 sentences), rationale_points (1–3 items), language (vi/ko/en).")
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    if OPENROUTER_SITE: headers["HTTP-Referer"] = OPENROUTER_SITE
    if OPENROUTER_APP:  headers["X-Title"] = OPENROUTER_APP
    url = f"{OPENROUTER_BASE.rstrip('/')}/chat/completions"

    def _try(payload, note: str):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=360)
            if r.status_code >= 400:
                try: raw = r.json()
                except Exception: raw = r.text[:2000]
                code = raw.get("error", {}).get("code") if isinstance(raw, dict) else None
                return {"error": _err("openrouter", OPENROUTER_MODEL, "http_error",
                                      "HTTP error", status=r.status_code, code=code, raw=raw)}
            data = r.json()
            txt = data["choices"][0]["message"]["content"]
            try:
                obj = json.loads(txt)
                if isinstance(obj, dict) and "explanation" in obj:
                    return obj
                return {"error": _err("openrouter", OPENROUTER_MODEL, "invalid_output",
                                      "Non-JSON or missing 'explanation'", raw=txt[:1000])}
            except Exception:
                return {"error": _err("openrouter", OPENROUTER_MODEL, "invalid_output",
                                      "Failed to parse JSON", raw=txt[:1000])}
        except requests.RequestException as rexc:
            return {"error": _err("openrouter", OPENROUTER_MODEL, "network_error", str(rexc))}

    payload1 = {"model": OPENROUTER_MODEL,
                "messages": [{"role": "system", "content": sysmsg},
                             {"role": "user", "content": prompt}],
                "temperature": 0.2,
                "response_format": {"type": "json_object"}}
    out = _try(payload1, "json_object")
    if "error" not in out: return out

    payload2 = {"model": OPENROUTER_MODEL,
                "messages": [{"role": "system", "content": sysmsg + " Return JSON only."},
                             {"role": "user", "content": prompt}],
                "temperature": 0.2}
    return _try(payload2, "loose")


# ---- Orchestrator ----
def call_llm_explainer(_unused_prompt: str, language: str, body: ExplainReq) -> Tuple[Dict[str, object], str, str, List[Dict[str, object]]]:
    errors: List[Dict[str, object]] = []
    log.info("LLM call order: local -> openrouter")

    # 1) LOCAL trước
    out_local = call_llm_local(body)
    if "error" not in out_local:
        text = str(out_local.get("explanation", "")).strip()
        if _is_valid_explain_text(text):
            return out_local, "local", out_local.get("model", "local"), errors
        errors.append(_err("local", "llm-service", "invalid_output", "Placeholder/invalid explanation", raw=text[:500]))
    else:
        errors.append(out_local["error"])

    # 2) OPENROUTER fallback: xin prompt từ llm-service để không trùng logic
    pr = fetch_prompt_from_local(body)
    if not pr:
        # safety net: nếu không lấy được prompt chuẩn, mới dùng builder cũ
        try:
            pr = format_prompt_for_llm(body)
        except Exception:
            pr = ""

    out_or = call_llm_openrouter(pr, language)
    if "error" not in out_or:
        text = str(out_or.get("explanation", "")).strip()
        if _is_valid_explain_text(text):
            return out_or, "openrouter", OPENROUTER_MODEL, errors
        errors.append(_err("openrouter", OPENROUTER_MODEL, "invalid_output", "Placeholder/invalid explanation", raw=text[:500]))
    else:
        errors.append(out_or["error"])

    fallback = {"vi":"Không thể sinh lời giải tự động từ các backend khả dụng.",
                "en":"Unable to generate explanation from available backends.",
                "ko":"사용 가능한 백엔드에서 해설을 생성할 수 없습니다."}
    return {"explanation": fallback.get((language or 'vi').lower(), fallback["vi"])}, "none", "none", errors


# ==================== PUBLIC SERVICE API ====================
def explain_question(body: ExplainReq, mode: str = "auto") -> ExplainResp:
    # NOTE: Chỉ tiền xử lý media ở client; prompt sẽ build ở llm-service
    base_payload = {
        "type": body.type, "section": body.section, "level": body.level,
        "title": body.title, "question": body.question,
        "dialogue": body.dialogue, "passage": body.passage,
        "options": body.options, "answer": body.answer,
    }

    qid = qid_for_question(base_payload)
    req_lang = (body.language or "vi").lower()

    log.info("Explain request | qid=%s level=%s cat=%s cau=%s lang=%s mode=%s passage=%s dialogue=%s",
             qid, body.level, body.category, body.cau, req_lang, mode, body.passage, body.dialogue)

    side = load_sidecar_explanation(qid, req_lang)
    if mode in ("auto", "cache") and side and _is_valid_explain_text(side.get("text","")):
        return ExplainResp(
            qid=qid,
            is_correct=(None if not body.user_answer else str(body.user_answer).strip().upper()==str(body.answer).strip().upper()),
            explanation=side.get("text") or "",
            updated=False,
            updated_path=str(_sidecar_path(qid)),
            explain_model=side.get("model"),
            errors=None,
        )
    if mode == "cache":
        return ExplainResp(qid=qid, is_correct=None, explanation="(Chưa có giải thích trong cache.)",
                           updated=False, updated_path=None, explain_model=None, errors=None)

    # --- Tiền xử lý media (ảnh -> mô tả, ưu tiên vision_records rồi RPC) ---
    body2 = _preprocess_media_in_body(body)

    print(body2)

    printable = {k: v for k, v in body2.dict().items() if k != "dialogue"}
    if isinstance(body2.options, list):
        printable["option"] = [f"{d.get('option','')}" for d in body2.options]

    # --- Gọi LLM orchestrator (local -> openrouter) ---
    llm_out, backend_used, model_used, err_list = call_llm_explainer("", req_lang, body2)

    explanation = ""
    rationale_points = None
    if isinstance(llm_out, dict):
        explanation = str(llm_out.get("explanation", "")).strip()
        rp = llm_out.get("rationale_points")
        if isinstance(rp, list): rationale_points = [str(x) for x in rp][:3]
    if not explanation:
        explanation = {"vi":"Hiện chưa có lời giải thích tự động.","en":"No automatic explanation.","ko":"자동 해설이 없습니다."}.get(req_lang, "Hiện chưa có lời giải thích tự động.")

    updated = False; updated_path = None
    if _is_valid_explain_text(explanation):
        entry = {"text": explanation, "points": (rationale_points or []), "lang": req_lang,
                 "model": model_used, "backend": backend_used, "ts": datetime.utcnow().isoformat() + "Z"}
        if body.use_sidecar or not body.source_jsonl:
            p = write_sidecar_lang(qid, req_lang, entry)
            updated, updated_path = True, str(p)
        else:
            updates = {"qid": qid, "explanations": {req_lang: entry}}
            p = Path(body.source_jsonl)
            updated = update_jsonl_inplace(p, qid, updates)
            updated_path = str(p) if updated else None

    is_correct = None
    if body.user_answer:
        is_correct = (str(body.user_answer).strip().upper() == str(body.answer).strip().upper())

    return ExplainResp(
        qid=qid,
        is_correct=is_correct,
        explanation=explanation,
        updated=updated,
        updated_path=updated_path,
        explain_model=(model_used if model_used != "none" else None),
        errors=(err_list or None),
    )
