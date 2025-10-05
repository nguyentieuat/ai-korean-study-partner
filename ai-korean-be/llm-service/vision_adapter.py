# llm-service/vision_adapter.py
import os, io, re, json, time, base64, hashlib, threading
from typing import List, Optional, Dict, Tuple
from datetime import datetime
from urllib.parse import urlparse

import requests
from PIL import Image

# ---------- logging ----------
import logging
VISION_LOG_LEVEL = os.getenv("VISION_LOG_LEVEL", "INFO").upper()
if not getattr(logging, "_vision_inited", False):
    logging.basicConfig(
        level=getattr(logging, VISION_LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)s | vision | %(message)s",
    )
    logging._vision_inited = True
log = logging.getLogger("vision")

def _short(s: Optional[str], n: int = 8) -> str:
    s = s or ""
    return s[:n]

def _ellipsize(s: Optional[str], n: int = 120) -> str:
    s = s or ""
    return s if len(s) <= n else (s[:n] + "…")

def _tag_src(src: str) -> str:
    """Rút gọn để log: không in cả data URL / full path dài."""
    if not src:
        return "<empty>"
    if src.startswith("data:image/"):
        return f"data:len={len(src)}"
    if src.startswith("http://") or src.startswith("https://"):
        u = urlparse(src)
        base = os.path.basename(u.path) or "/"
        host = u.netloc.split(":")[0]
        tail = base[-30:]
        return f"url://{host}/{tail}"
    return f"path:{os.path.basename(src)}"

# ---------- config ----------
OLLAMA = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
VISION_MODEL = os.getenv("VISION_MODEL", "moondream")
VISION_CACHE_DIR = os.getenv("VISION_CACHE_DIR", "data/vision_cache")
USE_OLLAMA_ON_MISS = os.getenv("USE_OLLAMA_ON_MISS", "true").lower() in {"1","true","yes","y"}
os.makedirs(VISION_CACHE_DIR, exist_ok=True)

_lock = threading.Lock()  # tránh đua ghi file cache

# ---------- utils ----------
def _sha1(b: bytes) -> str:
    h = hashlib.sha1(); h.update(b); return h.hexdigest()

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _imgkey_from_b64(b64: str) -> str:
    """Cache key: sha1(base64_bytes)."""
    return _sha1(b64.encode("utf-8"))

def _cache_path(key: str) -> str:
    # phẳng: không subfolder
    return os.path.join(VISION_CACHE_DIR, f"{key}.json")

def _cache_read(key: str) -> Optional[Dict]:
    p = _cache_path(key)
    if not os.path.exists(p):
        log.debug("CACHE_READ miss key=%s path=%s", _short(key), p)
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        langs = sorted(list((data.get("langs") or {}).keys()))
        log.debug("CACHE_READ hit  key=%s langs=%s", _short(key), langs)
        return data
    except Exception as e:
        log.warning("CACHE_READ error key=%s err=%s", _short(key), e)
        return None

def _cache_write_merge_lang(key: str, lang: str, text: str, backend: str):
    """Ghi/merge 1 ngôn ngữ vào file {key}.json với format chuẩn."""
    p = _cache_path(key)
    data = {"langs": {}, "meta": {}}
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                data.update(loaded)
        except Exception as e:
            log.warning("CACHE_WRITE parse-old failed key=%s err=%s", _short(key), e)

    data["langs"] = dict(data.get("langs") or {})
    data["langs"][lang] = text or ""
    meta = dict(data.get("meta") or {})
    meta["ts"] = datetime.utcnow().isoformat() + "Z"
    meta["backend"] = backend
    data["meta"] = meta

    tmp = p + ".tmp"
    try:
        with _lock:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, p)
        log.info(
            "CACHE_WRITE key=%s lang=%s backend=%s text_len=%d",
            _short(key), lang, backend, len(text or "")
        )
    except Exception as e:
        log.error("CACHE_WRITE error key=%s err=%s", _short(key), e)

# ---------- two-block helpers ----------
_TEXT_RE = re.compile(r"\[TEXT\](.*?)\[/TEXT\]", flags=re.S | re.I)
_DESC_RE = re.compile(r"\[DESC\](.*?)\[/DESC\]", flags=re.S | re.I)

def _split_blocks(s: str) -> Tuple[str, str]:
    mT = _TEXT_RE.search(s or ""); mD = _DESC_RE.search(s or "")
    t = (mT.group(1).strip() if mT else "")
    d = (mD.group(1).strip() if mD else "")
    return t, d

def _pack_blocks(txt: str, desc: str) -> str:
    txt = (txt or "").strip()
    desc = (desc or "").strip()
    return f"[TEXT]\n{txt}\n[/TEXT]\n\n[DESC]\n{desc}\n[/DESC]".strip()

# ---------- NLLB-200 translator (block-aware) ----------
_NLLB = {
    "model_id": "facebook/nllb-200-distilled-600M",
    "tok": None,
    "model": None,
    "ready": False,
    "codes": {"en": "eng_Latn", "ko": "kor_Hang", "vi": "vie_Latn"},
}

def _nllb_lazy_init():
    if _NLLB["ready"]:
        return True
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch  # device check only
        tok = AutoTokenizer.from_pretrained(_NLLB["model_id"])
        model = AutoModelForSeq2SeqLM.from_pretrained(_NLLB["model_id"])
        try:
            if torch.cuda.is_available():
                model = model.to("cuda")
        except Exception:
            pass
        _NLLB["tok"] = tok
        _NLLB["model"] = model
        _NLLB["ready"] = True
        log.info("NLLB ready: %s", _NLLB["model_id"])
        return True
    except Exception as e:
        log.warning("NLLB init failed: %s", e)
        _NLLB["ready"] = False
        return False

def _get_bos_id(tok, lang_code: str) -> int:
    # 1) slow tokenizer path
    if hasattr(tok, "lang_code_to_id"):
        return tok.lang_code_to_id[lang_code]
    # 2) fast tokenizer: try variants
    for cand in (f"__{lang_code}__", f"<2{lang_code}>", lang_code):
        tid = tok.convert_tokens_to_ids(cand)
        if tid is not None and tid != getattr(tok, "unk_token_id", None):
            return tid
    # 3) last resort
    bos = getattr(tok, "bos_token_id", None)
    if bos is not None:
        return bos
    raise ValueError(f"NLLB: cannot find BOS id for {lang_code}")

def _nllb_translate(text: str, src: str, tgt: str) -> str:
    """
    Plain text translation using NLLB; returns original on any error.
    src/tgt: 'en'|'ko'|'vi'
    """
    if not text:
        return text
    if src == tgt:
        return text
    if not _nllb_lazy_init():
        return text

    tok = _NLLB["tok"]; model = _NLLB["model"]
    src_code = _NLLB["codes"].get((src or "en").lower(), "eng_Latn")
    tgt_code = _NLLB["codes"].get((tgt or "ko").lower(), "kor_Hang")
    try:
        tok.src_lang = src_code
        enc = tok(text, return_tensors="pt")
        try:
            dev = model.device
            for k in enc:
                enc[k] = enc[k].to(dev)
        except Exception:
            pass
        bos_id = _get_bos_id(tok, tgt_code)
        gen = model.generate(**enc, forced_bos_token_id=bos_id, max_new_tokens=512)
        out = tok.batch_decode(gen, skip_special_tokens=True)[0].strip()
        return out or text
    except Exception as e:
        log.warning("NLLB translate fallback (%s->%s): %s", src, tgt, e)
        return text

def _mt_translate_blocks(en_blob: str, tgt: str) -> str:
    """
    Translate only [DESC] from EN to tgt; keep [TEXT] verbatim.
    If en_blob has no blocks, treat whole as DESC.
    """
    en_t, en_d = _split_blocks(en_blob)
    if not en_t and not en_d:
        tr_d = _nllb_translate(en_blob, "en", tgt)  # whole as desc
        return _pack_blocks("", tr_d)
    tr_d = _nllb_translate(en_d, "en", tgt) if en_d else ""
    return _pack_blocks(en_t, tr_d)

# ---------- Prompts (aligned with warm_vision_images_kaggle.py) ----------
def _prompt_two_blocks() -> str:
    return (
        "TASK: Return TWO blocks.\n"
        "Block 1 (TEXT): STRICTLY transcribe ALL visible text VERBATIM. Keep line breaks.\n"
        "Block 2 (DESC): Give a concise 1–2 sentence description of non-text content (icons, layout, scene).\n"
        "Rules:\n"
        "- NO extra commentary or language labels.\n"
        "- If no text exists, put <NO_TEXT> in Block 1.\n"
        "Format EXACTLY:\n"
        "[TEXT]\n<text here>\n[/TEXT]\n\n[DESC]\n<description here>\n[/DESC]"
    )

def _prompt_chart_blocks() -> str:
    return (
        "You are a data-visualization interpreter.\n"
        "IF the image is a chart/graph/plot/table:\n"
        "1) Identify chart type; 2) Summarize key trend in 1–2 sentences; "
        "3) DO NOT restate every label.\n"
        "Output ONLY the two blocks format:\n"
        "[TEXT]\n<verbatim text if present>\n[/TEXT]\n\n[DESC]\n<chart meaning summary>\n[/DESC]"
    )

# ---------- Ollama call ----------
def _lang_name(lang: str) -> str:
    return {"ko": "한국어", "vi": "Tiếng Việt", "en": "English"}.get((lang or "ko").lower(), "한국어")

def _call_ollama_one(
    b64_png: str,
    lang: str,
    instruction: str = "",
    model_used: Optional[str] = None,
    temperature: float = 0.1,
    chart_aware: bool = False,
) -> str:
    """
    Gọi 1 ảnh (base64) -> text theo chuẩn 2 block.
    - Luôn yêu cầu format EXACT: [TEXT]..[/TEXT] + [DESC]..[/DESC]
    - Nếu chart_aware=True: dùng prompt giải thích biểu đồ.
    - Block 1 (TEXT) phải verbatim; ngôn ngữ đầu ra áp cho Block 2 (DESC).
    """
    lang_line = f"Write Block 2 (DESC) in {_lang_name(lang)}. Keep Block 1 (TEXT) verbatim as in the image."

    base_prompt = _prompt_chart_blocks() if chart_aware else _prompt_two_blocks()
    final_instruction = base_prompt
    if instruction:
        final_instruction += "\n\n" + instruction

    payload = {
        "model": model_used or VISION_MODEL,
        "messages": [
            {"role": "system", "content": "You are a precise vision-to-text assistant."},
            {"role": "user", "content": f"{lang_line}\n\n{final_instruction}", "images": [b64_png]},
        ],
        "options": {"temperature": temperature},
        "stream": False,
    }
    try:
        log.debug(
            "OLLAMA_CALL model=%s lang=%s b64_len=%d",
            model_used or VISION_MODEL, lang, len(b64_png or "")
        )
        r = requests.post(f"{OLLAMA}/api/chat", json=payload, timeout=360)
        r.raise_for_status()
        data = r.json()
        txt = (data.get("message", {}) or {}).get("content", "").strip()
        log.info(
            "OLLAMA_OK   model=%s lang=%s out_len=%d",
            model_used or VISION_MODEL, lang, len(txt or "")
        )
        return txt
    except Exception as e:
        log.warning("OLLAMA_ERR model=%s lang=%s err=%s", model_used or VISION_MODEL, lang, e)
        return ""

# ---------- Public: cached (base64 batch) ----------
def vision_to_text_cached_b64(
    images_b64: List[str],
    instruction: str = "",
    lang: str = "ko",
    model: Optional[str] = None,
    temperature: float = 0.1,
) -> List[str]:
    """
    Nhận danh sách base64 (không data URL), trả text theo đúng thứ tự.
    Chính sách:
      - Cache theo ảnh: data/vision_cache/<sha1(base64)>.json
      - Ưu tiên trả KO. Nếu chưa có KO nhưng có EN -> dịch EN→lang (chỉ DESC), ghi bù.
      - Nếu miss hoàn toàn -> (tùy cờ) gọi Ollama sinh EN với 2-block rồi dịch -> ghi cache.
    """
    log.info(
        "VISION_BATCH start n=%d lang=%s model=%s",
        len(images_b64), lang, model or VISION_MODEL
    )
    out: List[str] = []

    for idx, b64 in enumerate(images_b64):
        b64 = (b64 or "").strip()
        if not b64:
            log.warning("VISION_ITEM[%d] empty base64 -> skip", idx)
            out.append("")
            continue

        key = _imgkey_from_b64(b64)
        hit = _cache_read(key)
        langs_map = (hit or {}).get("langs") or {}
        meta_backend = ((hit or {}).get("meta") or {}).get("backend") or (model or VISION_MODEL)

        # 1) Có đúng ngôn ngữ yêu cầu?
        if isinstance(langs_map.get(lang), str) and langs_map[lang].strip():
            txt = langs_map[lang]
            out.append(txt)
            log.info(
                "VISION_ITEM[%d] key=%s HIT[%s] len=%d",
                idx, _short(key), lang, len(txt or "")
            )
            continue

        # 2) Có EN? -> dịch EN→lang (block-aware)
        if lang != "en" and isinstance(langs_map.get("en"), str) and langs_map["en"].strip():
            en_blob = langs_map["en"]
            try:
                has_blocks = bool(_TEXT_RE.search(en_blob) or _DESC_RE.search(en_blob))
                if has_blocks:
                    tr_blob = _mt_translate_blocks(en_blob, lang)
                else:
                    tr_desc = _nllb_translate(en_blob, "en", lang)
                    tr_blob = _pack_blocks("", tr_desc)
            except Exception as e:
                log.warning("BLOCK-MT fallback: %s", e)
                tr_text = _nllb_translate(en_blob, "en", lang)
                tr_blob = _pack_blocks("", tr_text)

            out.append(tr_blob)
            log.info(
                "VISION_ITEM[%d] key=%s HIT[en]->NLLB[%s] len_en=%d len_tr=%d",
                idx, _short(key), lang, len(en_blob or ""), len(tr_blob or "")
            )
            try:
                _cache_write_merge_lang(key, lang, tr_blob, backend=meta_backend + "+nllb")
            except Exception as e:
                log.warning("CACHE_WRITE fail key=%s lang=%s err=%s", _short(key), lang, e)
            continue

        # 3) Miss -> gọi Ollama sinh EN dạng 2-block rồi dịch
        log.info(
            "VISION_ITEM[%d] key=%s MISS -> %s",
            idx, _short(key), "OLLAMA_EN+NLLB" if USE_OLLAMA_ON_MISS else "EMPTY"
        )
        txt = ""
        if USE_OLLAMA_ON_MISS:
            # Heuristic rất nhẹ để bật chart-aware nếu caller ám chỉ
            chart_aware = "chart" in (instruction or "").lower()
            base_instr = instruction or _prompt_two_blocks()  # ép 2 block mặc định
            en_tmp = _call_ollama_one(
                b64_png=b64,
                lang="en",
                instruction=base_instr,
                model_used=model,
                temperature=temperature,
                chart_aware=chart_aware,
            )
            if en_tmp:
                has_blocks = bool(_TEXT_RE.search(en_tmp) or _DESC_RE.search(en_tmp))
                if not has_blocks:
                    en_tmp = _pack_blocks("", en_tmp)
                if lang == "en":
                    txt = en_tmp
                else:
                    txt = _mt_translate_blocks(en_tmp, lang)
                # ghi cache
                try:
                    _cache_write_merge_lang(key, "en", en_tmp, backend=(model or VISION_MODEL))
                    if lang != "en":
                        _cache_write_merge_lang(key, lang, txt, backend=(model or VISION_MODEL) + "+nllb")
                except Exception as e:
                    log.warning("CACHE_WRITE fail key=%s err=%s", _short(key), e)

        out.append(txt or "")

    log.info("VISION_BATCH done n=%d", len(images_b64))
    return out

_B64_RE = re.compile(r'^[A-Za-z0-9+/=\r\n]+$')

def looks_like_b64(s: str, min_len: int = 64) -> bool:
    if not s or not _B64_RE.match(s):
        return False
    t = s.strip()
    return len(t) >= min_len and (len(t) % 4 == 0)

# ---------- Back-compat (optional) ----------
def vision_to_text_cached(
    image_sources: List[str],
    instruction: str,
    lang: str = "ko",
    model: Optional[str] = None,
    temperature: float = 0.1,
) -> Dict[str, str]:
    """
    Giữ lại để tương thích: nhận URL/dataURL/path/b64-literal -> chuyển thành base64 nếu đọc được,
    rồi dùng vision_to_text_cached_b64.
    """
    srcs = []
    raw_keys = []  # log/đối chiếu

    for s in image_sources:
        if not s:
            srcs.append("")
            raw_keys.append(None)
            continue

        if s.startswith("data:image/"):
            try:
                b64 = s.split(",", 1)[1]
                raw = base64.b64decode(b64)
                key = _sha1(base64.b64encode(raw))  # sha1(base64(original_bytes))
                srcs.append(b64)          # giữ nguyên base64 gốc, KHÔNG convert PNG
                raw_keys.append(key)
                log.debug("SRC data-url -> b64 len=%d tag=%s", len(b64), _tag_src(s))
            except Exception as e:
                log.warning("SRC data-url parse fail tag=%s err=%s", _tag_src(s), e)
                srcs.append("")
                raw_keys.append(None)

        elif s.startswith("http://") or s.startswith("https://"):
            try:
                r = requests.get(s, timeout=20)
                r.raise_for_status()
                raw = r.content
                key = _sha1(base64.b64encode(raw))   # sha1(base64(original_bytes))
                b64 = base64.b64encode(raw).decode("utf-8")  # KHÔNG re-encode PNG
                srcs.append(b64)
                raw_keys.append(key)
                log.debug("SRC url -> b64 len=%d tag=%s", len(b64), _tag_src(s))
            except Exception as e:
                log.warning("SRC url load fail tag=%s err=%s", _tag_src(s), e)
                srcs.append("")
                raw_keys.append(None)

        else:
            if not os.path.exists(s) and looks_like_b64(s):
                try:
                    raw = base64.b64decode(s, validate=True)
                    key = _sha1(base64.b64encode(raw))          # sha1(base64(raw_bytes))
                    b64 = s.strip()                              # dùng chính base64 gốc
                    srcs.append(b64)
                    raw_keys.append(key)
                    log.debug("SRC b64-literal detected len=%d tag=%s", len(b64), _tag_src(s))
                except Exception as e:
                    log.warning("SRC b64-literal decode fail tag=%s err=%s", _tag_src(s), e)
                    srcs.append("")
                    raw_keys.append(None)
            else:
                try:
                    raw = open(s, "rb").read()                  # RAW bytes từ file
                    key = _sha1(base64.b64encode(raw))          # sha1(base64(raw_bytes))
                    b64 = base64.b64encode(raw).decode("utf-8") # KHÔNG re-encode PNG
                    srcs.append(b64)
                    raw_keys.append(key)
                    log.debug("SRC path -> b64 len=%d tag=%s", len(b64), _tag_src(s))
                except Exception as e:
                    log.warning("SRC path load fail tag=%s err=%s", _tag_src(s), e)
                    srcs.append("")
                    raw_keys.append(None)

    texts = vision_to_text_cached_b64(
        srcs, instruction=instruction, lang=lang, model=model, temperature=temperature
    )

    # log mapping src -> key (rút gọn) để đối chiếu
    for s, b64, key in zip(image_sources, srcs, raw_keys):
        log.info("MAP src=%s key=%s", _tag_src(s), _short(key) if key else "<no-key>")

    return {src: txt for src, txt in zip(image_sources, texts)}
