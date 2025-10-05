#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Warm images -> text (Hybrid OCR/Caption) for TOPIK assets (Kaggle-ready)

- Key cache: sha1(base64(raw_bytes))  (KHÔNG re-encode PNG trước khi hash)
- Output format (per language):
  [TEXT]
  <verbatim extracted text (OCR or LLaVA TEXT block)>
  [/TEXT]

  [DESC]
  <short description / chart insight>
  [/DESC]

- Routing:
  PURE_TEXT -> OCR as TEXT, DESC empty
  BANNER    -> LLaVA two-block, fallback OCR fills TEXT if needed
  CHART     -> OCR for TEXT + LLaVA (chart-prompt) for DESC
  SCENE     -> LLaVA for DESC only

Defaults:
  pivot = "en" (internal), langs="ko" (TOPIK yêu cầu Hàn)
"""

import os, io, re, json, base64, hashlib, argparse, warnings
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime

import pandas as pd
from PIL import Image
from tqdm import tqdm

# --------------------
# Config & logging
# --------------------
import logging
LOG_LEVEL = os.getenv("WARM_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | warm | %(message)s",
)
log = logging.getLogger("warm")

# Silence HF transformers noise + TrOCR warnings
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()  # only show errors from transformers
except Exception:
    pass

# Hide specific TrOCR init warnings
warnings.filterwarnings(
    "ignore",
    message=r".*VisionEncoderDecoderModel were not initialized.*"
)
warnings.filterwarnings(
    "ignore",
    message=r"You should probably TRAIN this model on a down-stream task.*"
)

HF_HOME = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("BNB_CUDA_VERSION", "120")  # Kaggle CUDA 12.x

# --------------------
# Utils
# --------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

def sha1_of_b64str(b64: str) -> str:
    return hashlib.sha1(b64.encode("utf-8")).hexdigest()

def key_from_raw_bytes(raw: bytes) -> str:
    b64 = base64.b64encode(raw).decode("utf-8")
    return sha1_of_b64str(b64)

def list_images(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]

def read_raw_bytes(p: Path) -> Optional[bytes]:
    try:
        return p.read_bytes()
    except Exception as e:
        log.warning("read fail: %s (%s)", p, e)
        return None

def chunked(lst: List, bs: int) -> List[List]:
    return [lst[i:i+bs] for i in range(0, len(lst), bs)]

def short(s: Optional[str], n: int = 8) -> str:
    s = s or ""
    return s[:n]

def pack_text_desc(txt: str, desc: str) -> str:
    txt = (txt or "").strip()
    desc = (desc or "").strip()
    return f"[TEXT]\n{txt}\n[/TEXT]\n\n[DESC]\n{desc}\n[/DESC]".strip()

# --------------------
# Simple path meta parsing (optional)
# --------------------
def parse_meta_from_path(p: Path) -> Dict[str, Optional[str]]:
    parts = [seg.lower() for seg in p.parts]
    level = None; section = None
    for i, seg in enumerate(parts):
        if seg in ("topik1", "topik2"):
            level = seg
            if i + 1 < len(parts):
                sec = parts[i+1]
                if sec.startswith("listen"): section = "listen"
                elif sec.startswith("read"): section = "reading"
            break
    typ = "nghe" if section == "listen" else ("doc" if section == "reading" else None)

    stem = p.stem
    tokens = stem.split("_")
    de = cau = opt = None
    if len(tokens) >= 3:
        last = tokens[-1]
        if re.fullmatch(r"[A-Za-z]", last):
            opt = last.lower()
            m = re.search(r"(\d+)", tokens[-2])
            if m: cau = m.group(1)
            de = "_".join(tokens[:-2])
        else:
            m2 = re.search(r"(\d+)$", last)
            if m2:
                cau = m2.group(1)
                de = "_".join(tokens[:-1])
            else:
                de = stem
    else:
        de = stem

    return {"level": level, "section": section, "type": typ,
            "de": de, "cau": cau, "option": opt, "stem": stem}

def build_key(meta: Dict[str, Optional[str]], lang: str) -> str:
    return "type={}|level={}|cau={}|opt={}|lang={}".format(
        meta.get("type") or "", meta.get("level") or "",
        meta.get("cau") or "", meta.get("option") or "", lang
    )

# --------------------
# Cache (flat files)
# --------------------
class VisionCache:
    """
    {cache_dir}/{sha1(base64(raw_bytes))}.json
    {
      "langs": {"en": "...", "ko": "...", ...},
      "meta": {"ts": "...Z", "backend": "llava:...", "route": "banner" }
    }
    """
    def __init__(self, cache_dir: Path):
        self.dir = cache_dir; self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> Optional[Dict]:
        p = self._path(key)
        if not p.exists(): return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("cache parse fail key=%s err=%s", short(key), e)
            return None

    def put_langs(self, key: str, langs: Dict[str, str], meta: Dict[str, str]):
        p = self._path(key)
        data = {"langs": {}, "meta": {}}
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                data = {"langs": {}, "meta": {}}
        data["langs"] = dict(data.get("langs") or {})
        data["langs"].update(langs or {})
        meta0 = dict(data.get("meta") or {})
        meta0.update(meta or {})
        data["meta"] = meta0
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, p)

# --------------------
# Translation (NLLB-200) — robust, quality EN<->KO/VI
# --------------------
# --------------------
# Translation (NLLB-200) — robust, works with fast/slow tokenizer
# --------------------
class MTGraph:
    """
    facebook/nllb-200-distilled-600M
    - Language codes: eng_Latn, kor_Hang, vie_Latn
    - Falls back to original text if model/tokenizer not available.
    """
    def __init__(self, device: str = "cpu"):
        self.enabled = True
        self.device = device
        self.model_id = "facebook/nllb-200-distilled-600M"
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            self.AutoTokenizer = AutoTokenizer
            self.AutoModelForSeq2SeqLM = AutoModelForSeq2SeqLM

            self.tok = self.AutoTokenizer.from_pretrained(self.model_id)  # fast OK
            self.model = self.AutoModelForSeq2SeqLM.from_pretrained(self.model_id)

            if device == "cuda":
                import torch
                self.model = self.model.to("cuda")
        except Exception as e:
            log.warning("[MT] init disabled: %s", e)
            self.enabled = False
            self.tok = None
            self.model = None

        # NLLB language code map
        self.code = {"en": "eng_Latn", "ko": "kor_Hang", "vi": "vie_Latn"}

    @staticmethod
    def _norm_lang(s: str) -> str:
        s = (s or "").lower()
        if s.startswith("en"): return "en"
        if s.startswith("vi"): return "vi"
        if s.startswith("ko"): return "ko"
        return "en"

    def _get_bos_id(self, lang_code: str) -> int:
        """
        Handle both slow and fast tokenizers:
        - slow: has tok.lang_code_to_id
        - fast: try __{code}__, <2{code}>, {code}
        """
        tok = self.tok
        if hasattr(tok, "lang_code_to_id"):
            return tok.lang_code_to_id[lang_code]

        # fast tokenizer path
        for cand in (f"__{lang_code}__", f"<2{lang_code}>", lang_code):
            tid = tok.convert_tokens_to_ids(cand)
            if tid is not None and tid != getattr(tok, "unk_token_id", None):
                return tid

        # last resort: BOS token id (may reduce quality but keeps it running)
        bos = getattr(tok, "bos_token_id", None)
        if bos is not None:
            return bos
        raise ValueError(f"[MT] Cannot find BOS id for {lang_code}")

    def translate_one(self, text: str, src: str, tgt: str) -> str:
        if not self.enabled or not text:
            return text

        src = self._norm_lang(src); tgt = self._norm_lang(tgt)
        if src == tgt:
            return text

        try:
            src_code = self.code.get(src, "eng_Latn")
            tgt_code = self.code.get(tgt, "kor_Hang")

            # set source language for tokenizer
            self.tok.src_lang = src_code

            # tokenize & move to device
            enc = self.tok(text, return_tensors="pt")
            if self.device == "cuda":
                for k in enc:
                    enc[k] = enc[k].to(self.model.device)

            # get BOS id for target lang (works for fast/slow)
            bos_id = self._get_bos_id(tgt_code)

            gen = self.model.generate(
                **enc,
                forced_bos_token_id=bos_id,
                max_new_tokens=512
            )
            out = self.tok.batch_decode(gen, skip_special_tokens=True)[0].strip()
            return out or text
        except Exception as e:
            log.warning("[MT] translate fallback (%s->%s) due to: %s", src, tgt, e)
            return text

    def translate_batch(self, texts: List[str], src: str, tgt: str) -> List[str]:
        return [self.translate_one(t, src, tgt) for t in texts]

# --------------------
# Backends
# --------------------
class BackendBase:
    def describe_batch(self, pil_imgs: List[Image.Image], lang: str) -> List[str]:
        raise NotImplementedError

class BackendTrOCR(BackendBase):
    def __init__(self, device="cpu"):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        self.proc = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
        self.device = device
        if device == "cuda":
            import torch
            self.model = self.model.to("cuda")
        self.mode = "trocr-base-printed"

    def describe_batch(self, pil_imgs: List[Image.Image], lang: str) -> List[str]:
        import torch
        with torch.no_grad():
            pixel = self.proc(images=pil_imgs, return_tensors="pt").pixel_values
            if self.device == "cuda":
                pixel = pixel.to("cuda")
            out_ids = self.model.generate(pixel, max_length=2048)
            texts = self.proc.batch_decode(out_ids, skip_special_tokens=True)
        return [t.strip() for t in texts]

class BackendLLaVA(BackendBase):
    """LLaVA 1.5 7B HF — two-block output [TEXT]/[DESC] with chart-aware instruction."""
    def __init__(self, device="cuda", quantize: str = "8bit", max_new_tokens: int = 192, classify_only: bool=False):
        import torch
        from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

        model_id = os.getenv("LLAVA_MODEL_ID", "llava-hf/llava-1.5-7b-hf")

        # Force slow tokenizer + trust remote code
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=False,
        )

        # Quantization config (no deprecated flags)
        bnb_config = None
        if quantize == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        elif quantize == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

        kwargs = {
            "device_map": "auto",
            "quantization_config": bnb_config,
            "trust_remote_code": True,
        }
        if bnb_config is None:
            kwargs["torch_dtype"] = torch.float16  # FP16

        self.model = LlavaForConditionalGeneration.from_pretrained(model_id, **kwargs)
        self.mode = f"llava:{Path(model_id).name}{'+cls' if classify_only else ''}"
        self.max_new_tokens = max_new_tokens
        self.classify_only = classify_only

    def _prompt_blocks(self) -> str:
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

    def _prompt_chart(self) -> str:
        return (
            "You are a data-visualization interpreter.\n"
            "IF the image is a chart/graph/plot/table:\n"
            "1) Identify chart type; 2) Summarize key trend in 1–2 sentences; "
            "3) DO NOT restate every label.\n"
            "Output ONLY the two blocks format:\n"
            "[TEXT]\n<verbatim text if present>\n[/TEXT]\n\n[DESC]\n<chart meaning summary>\n[/DESC]"
        )

    def _prompt_classify(self) -> str:
        return (
            "Classify the image into exactly one of: PURE_TEXT, BANNER, CHART, SCENE.\n"
            "- PURE_TEXT: mostly printed text.\n"
            "- BANNER: mix of text and graphics/icons.\n"
            "- CHART: charts/plots/tables with axes/bars/lines/pie.\n"
            "- SCENE: photo/illustration where text is not the main content.\n"
            "Reply with only the label in UPPERCASE."
        )

    def describe_batch(self, pil_imgs: List[Image.Image], lang: str) -> List[str]:
        import torch
        prompt = self._prompt_classify() if self.classify_only else self._prompt_blocks()
        chats = [[{"role": "user", "content": [{"type": "text", "text": prompt},
                                               {"type": "image"}]}] for _ in pil_imgs]
        texts = [self.processor.apply_chat_template(c, add_generation_prompt=True)
                 for c in chats]
        inputs = self.processor(text=texts, images=pil_imgs, return_tensors="pt", padding=True)

        # move inputs to same device as model
        try:
            dev = self.model.device
        except Exception:
            dev = next(self.model.parameters()).device
        inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}

        with torch.cuda.amp.autocast(enabled=(hasattr(dev, "type") and dev.type == "cuda")):
            gen = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)

        # decode per row using attn lengths if available
        outs = []
        attn = inputs.get("attention_mask", None)
        if attn is None:
            outs = self.processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        else:
            lengths = attn.sum(dim=1)
            for i in range(gen.shape[0]):
                start = int(lengths[i].item())
                out_ids = gen[i, start:].unsqueeze(0)
                txt = self.processor.batch_decode(
                    out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                outs.append(txt.strip())
        return [o.strip() for o in outs]

# --------------------
# Routing heuristics
# --------------------
from enum import Enum
class VisionRoute(str, Enum):
    PURE_TEXT="pure_text"; BANNER="banner"; CHART="chart"; SCENE="scene"

KW_CHART = {"chart","graph","plot","axis","axes","legend","table","grid","scatter",
            "bar","line","pie","histogram","x-axis","y-axis","%", "rate", "index",
            "그래프","도표","표","축","범례","비율","지표"}
KW_BANNER = {"sale","event","discount","www","http","qr","coupon","hotline","register","notice",
             "공지","이벤트","할인","행사","등록","안내","문의","쿠폰","포스터","홍보"}

def decide_route(ocr_text: str, llava_label: str="") -> VisionRoute:
    t=(ocr_text or "").lower().strip()
    line_count=t.count("\n")+1 if t else 0
    digits=sum(ch.isdigit() for ch in t)
    alnum=sum(ch.isalnum() for ch in t) or 1
    digit_ratio=digits/alnum
    label = (llava_label or "").strip().upper()

    # 1) trust model label if it’s one of the four (lookup by NAME)
    if label in {"CHART","PURE_TEXT","BANNER","SCENE"}:
        try:
            return VisionRoute[label]
        except KeyError:
            pass

    # 2) strong heuristics
    if any(k in t for k in KW_CHART) or (digit_ratio>=0.25 and line_count>=3):
        return VisionRoute.CHART
    if any(k in t for k in KW_BANNER):
        return VisionRoute.BANNER
    if len(t)>=80 and not any(k in t for k in KW_CHART):
        return VisionRoute.PURE_TEXT

    # 3) fallback
    if len(t) <= 10:
        return VisionRoute.SCENE
    return VisionRoute.BANNER


_TEXT_RE = re.compile(r"\[TEXT\](.*?)\[/TEXT\]", flags=re.S|re.I)
_DESC_RE = re.compile(r"\[DESC\](.*?)\[/DESC\]", flags=re.S|re.I)
def split_blocks(s: str) -> Tuple[str,str]:
    mT=_TEXT_RE.search(s or ""); mD=_DESC_RE.search(s or "")
    t=(mT.group(1).strip() if mT else "")
    d=(mD.group(1).strip() if mD else "")
    return t,d

_GENERIC = (
    "the image is", "a foreign language text", "black background",
    "the picture shows", "text in a foreign language"
)
def looks_generic(s: str) -> bool:
    x=(s or "").strip().lower()
    if not x: return True
    if x=="<no_text>": return True
    if any(k in x for k in _GENERIC): return True
    if ("\n" not in x) and (len(x)<80) and x.endswith("."): return True
    return False

def looks_bad_ko(s: str) -> bool:
    """Heuristic: KO phải có đủ ký tự Hangul; nếu Latin quá nhiều hoặc Hangul quá ít -> xem là rác."""
    if not s or not s.strip(): 
        return True
    # Đếm Hangul (Jamo + Hangul syllables)
    h = 0
    for ch in s:
        o = ord(ch)
        if (0xAC00 <= o <= 0xD7A3) or (0x1100 <= o <= 0x11FF) or (0x3130 <= o <= 0x318F):
            h += 1
    latin = sum(1 for ch in s if ('a' <= ch <= 'z') or ('A' <= ch <= 'Z'))
    # Ngưỡng đơn giản: cần >=10 ký tự Hangul và Latin không vượt quá Hangul
    return latin > h
# --------------------
# Pipeline runner
# --------------------
def run_pipeline(route: VisionRoute,
                 pil_img: Image.Image,
                 llava_blocks: BackendLLaVA,
                 trocr: BackendTrOCR,
                 mt: MTGraph,
                 target_langs: List[str]) -> Dict[str,str]:
    """
    Return dict(lang -> two-block string)
    Only translate DESC; keep TEXT verbatim.
    """
    results: Dict[str,str] = {}

    if route == VisionRoute.PURE_TEXT:
        en_txt = trocr.describe_batch([pil_img], "en")[0]
        results["en"] = pack_text_desc(en_txt, "")

    elif route == VisionRoute.CHART:
        en_txt = trocr.describe_batch([pil_img], "en")[0]
        # *** reuse llava_blocks (no new model instantiation) ***
        desc_raw = llava_blocks.describe_batch([pil_img], "en")[0]
        _t,_d = split_blocks(desc_raw)
        desc = _d or desc_raw.strip()
        results["en"] = pack_text_desc(en_txt, desc)

    elif route == VisionRoute.BANNER:
        raw = llava_blocks.describe_batch([pil_img], "en")[0]
        txt, desc = split_blocks(raw)
        if not txt or looks_generic(txt):
            ocr_txt = trocr.describe_batch([pil_img], "en")[0]
            if ocr_txt and not looks_generic(ocr_txt):
                txt = ocr_txt
        results["en"] = pack_text_desc(txt, desc)

    else:  # SCENE
        raw = llava_blocks.describe_batch([pil_img], "en")[0]
        _t,_d = split_blocks(raw)
        desc = _d or raw.strip()
        results["en"] = pack_text_desc("", desc)

    # Translate only DESC
    en_t, en_d = split_blocks(results["en"])
    for lg in target_langs:
        if lg == "en": 
            continue
        tr_d = mt.translate_one(en_d, "en", lg) if en_d.strip() else ""
        # KO quality guard: nếu bản KO nhìn như rác -> fallback dùng EN
        if lg == "ko" and looks_bad_ko(tr_d):
            tr_d = en_d
        results[lg] = pack_text_desc(en_t, tr_d)

    return results

def torch_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

# --------------------
# Main process
# --------------------
def process(args):
    root = Path(args.root).resolve()
    assert root.exists(), f"Root not found: {root}"

    # device
    device = "cuda" if args.device == "cuda" else "cpu"
    if args.device == "auto":
        device = "cuda" if torch_cuda() else "cpu"

    # backends (single instances reused everywhere)
    trocr = BackendTrOCR(device=device)
    llava_blocks = BackendLLaVA(device=device, quantize=args.quantize, max_new_tokens=args.max_new_tokens)
    llava_cls = BackendLLaVA(device=device, quantize=args.quantize, max_new_tokens=8, classify_only=True)

    pivot = (args.pivot or "en").strip().lower()
    langs = [s.strip().lower() for s in (args.langs or "ko").split(",") if s.strip()]
    if pivot not in langs: langs = [pivot] + langs

    cache = VisionCache(Path(args.cache_dir))
    paths = list_images(root)

    log.info("Found %d images under %s", len(paths), root)
    log.info("Device=%s | pivot=%s | langs=%s", device, pivot, langs)
    log.info("Cache dir: %s", args.cache_dir)

    mt = MTGraph(device=device)
    rows = []

    # Preload: read raw, meta, key, cache
    items = []
    for p in paths:
        raw = read_raw_bytes(p)
        meta = parse_meta_from_path(p)
        if not raw:
            items.append((p, None, meta, "", {}))
            continue
        key = key_from_raw_bytes(raw)
        hit = cache.get(key) or {}
        items.append((p, raw, meta, key, hit.get("langs", {})))

    # Iterate in batches
    batches = chunked(items, args.batch_size)
    for batch in tqdm(batches, total=len(batches), desc="Processing", ncols=100):
        # Build PILs for items needing work (for routing & inference)
        pil_imgs = []
        pil_indices = []   # map j -> i in batch
        for i, (p, raw, meta, key, cached_langs) in enumerate(batch):
            if not raw: continue
            with Image.open(io.BytesIO(raw)) as im:
                pil_imgs.append(im.convert("RGB"))
                pil_indices.append(i)

        # quick classify labels (tiny LLaVA call)
        labels = []
        if pil_imgs:
            try:
                lbls = llava_cls.describe_batch(pil_imgs, "en")
                for s in lbls:
                    s = (s or "").strip().upper()
                    lab = "SCENE"
                    for cand in ("PURE_TEXT","BANNER","CHART","SCENE"):
                        if cand in s:
                            lab = cand; break
                    labels.append(lab)
            except Exception as e:
                log.warning("classify batch fail: %s", e)
                labels = [""] * len(pil_imgs)

        # Run per item
        for j, i in enumerate(pil_indices):
            p, raw, meta, key, cached_langs = batch[i]
            alias = f"{meta.get('level') or 'lvl'}_{meta.get('section') or 'sec'}_{meta.get('stem')}"

            # if we already have 'en' cached, reuse for routing (split to get TEXT length)
            cached_en = (cached_langs or {}).get("en", "")
            if cached_en:
                en_t, _ = split_blocks(cached_en)
                ocr_quick = en_t
            else:
                # cheap OCR to decide route (REUSE trocr instance)
                try:
                    with Image.open(io.BytesIO(raw)) as _im:
                        ocr_quick = trocr.describe_batch([_im.convert("RGB")], "en")[0]
                except Exception:
                    ocr_quick = ""

            ll = labels[j] if j < len(labels) else ""
            route = decide_route(ocr_quick, ll)

            # Compute multi-lang text via route
            try:
                result_langs = {}
                if cached_en:
                    # we already have en; just generate target langs from it
                    en_t, en_d = split_blocks(cached_en)
                    for lg in langs:
                        if lg == "en":
                            result_langs["en"] = cached_en
                        else:
                            tr_d = mt.translate_one(en_d, "en", lg) if en_d.strip() else ""
                            result_langs[lg] = pack_text_desc(en_t, tr_d)
                else:
                    # run the full pipeline for this image
                    with Image.open(io.BytesIO(raw)) as im:
                        en_and_targets = run_pipeline(route, im.convert("RGB"), llava_blocks, trocr, mt, langs)
                        if "en" not in en_and_targets:
                            en_and_targets["en"] = pack_text_desc(ocr_quick, "")
                        result_langs.update(en_and_targets)

                # Write cache (at least en; and the first requested output 'ko' if present)
                meta_info = {"ts": datetime.utcnow().isoformat() + "Z",
                             "backend": llava_blocks.mode,
                             "route": route.value}
                cache_payload = {}
                if "en" in result_langs and result_langs["en"].strip():
                    cache_payload["en"] = result_langs["en"]
                for lg in langs:
                    if lg == "en": continue
                    if lg in result_langs and result_langs[lg].strip():
                        cache_payload[lg] = result_langs[lg]
                        break  # write at least one target (ko default)
                if cache_payload:
                    cache.put_langs(key, cache_payload, meta_info)

                # Append rows (EN + requested langs)
                for lg in ([pivot] + [x for x in langs if x != pivot]):
                    txt = result_langs.get(lg, "")
                    rows.append({
                        "path": str(p),
                        "type": meta.get("type"),
                        "level": meta.get("level"),
                        "section": meta.get("section"),
                        "de": meta.get("de"),
                        "cau": meta.get("cau"),
                        "option": meta.get("option"),
                        "language": lg,
                        "key": build_key(meta, lg),
                        "alias": alias,
                        "text": txt,
                        "mode": llava_blocks.mode,
                        "status": "ok_cache" if cached_en else "ok_new",
                        "error": "",
                        "route": route.value,
                    })

            except Exception as e:
                for lg in ([pivot] + [x for x in langs if x != pivot]):
                    rows.append({
                        "path": str(p),
                        "type": meta.get("type"),
                        "level": meta.get("level"),
                        "section": meta.get("section"),
                        "de": meta.get("de"),
                        "cau": meta.get("cau"),
                        "option": meta.get("option"),
                        "language": lg,
                        "key": build_key(meta, lg),
                        "alias": alias,
                        "text": "",
                        "mode": llava_blocks.mode,
                        "status": "infer_fail",
                        "error": str(e),
                        "route": route.value,
                    })

    df = pd.DataFrame(rows)
    out = Path(args.report); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    log.info("Wrote report: %s (%d rows)", out, len(df))
    log.info("Cache dir   : %s", args.cache_dir)

def main():
    ap = argparse.ArgumentParser("Warm images -> text (hybrid OCR/Caption, Kaggle)")
    ap.add_argument("--root", required=True, help="Thư mục dữ liệu gốc (chứa topik1/topik2/...)")
    ap.add_argument("--report", default="./warm_report.csv", help="CSV output")
    ap.add_argument("--cache-dir", default="./data/vision_cache", help="Thư mục cache json (flat)")
    ap.add_argument("--langs", default=os.getenv("VISION_OUT_LANGS", "ko"),
                    help="Danh sách ngôn ngữ output, vd: 'ko,vi'")
    ap.add_argument("--pivot", default=os.getenv("VISION_PIVOT", "en"),
                    help="Ngôn ngữ pivot nội bộ (khuyên dùng 'en')")
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"], help="Thiết bị")
    ap.add_argument("--quantize", default="8bit", choices=["none","8bit","4bit"],
                    help="Cho LLaVA; 8bit là lựa chọn an toàn trên T4 16GB")
    ap.add_argument("--batch-size", type=int, default=8, help="Batch size ảnh (routing/infer)")
    ap.add_argument("--max-new-tokens", type=int, default=192, help="Giới hạn token sinh ra (LLaVA)")
    args = ap.parse_args()
    process(args)

if __name__ == "__main__":
    main()
