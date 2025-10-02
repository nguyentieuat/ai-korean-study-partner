# generate_question/utils.py
from __future__ import annotations
import json, hashlib, re
from typing import Dict, List, Optional

_SPEAKER_RE = re.compile(r"^(남자|여자|학생|선생님|교사|해설자|남|여)[:：]\s*(.+)$")

# ====== Tiny helpers ======

def norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()

def stable_id(obj: dict) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

# --- choices/answer normalization ---

def _clean_option_text(s: str) -> str:
    if not isinstance(s, str):
        return s
    return re.sub(r"\s*x\s*$", "", s.strip(), flags=re.IGNORECASE)

def normalize_choices(obj: dict) -> Dict[str, str]:
    src = obj.get("choices")
    if src is None:
        src = obj.get("options")
    if src is None:
        return {}
    if isinstance(src, dict):
        normed = {}
        for k, v in src.items():
            if isinstance(v, str):
                key = str(k).strip().upper()
                if len(key) == 1 and "A" <= key <= "Z":
                    normed[key] = v
        if not normed and all(str(k).isdigit() for k in src.keys()):
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for i, k in enumerate(sorted(src.keys(), key=lambda x: int(x))):
                if i < len(letters) and isinstance(src[k], str):
                    normed[letters[i]] = src[k]
        return normed
    if isinstance(src, list):
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return {letters[i]: s for i, s in enumerate(src) if isinstance(s, str)}
    return {}

def normalize_answer(obj: dict, choices: Dict[str, str]) -> str:
    ans = obj.get("answer")
    if ans is None:
        return "A" if choices else ""
    if isinstance(ans, str):
        up = ans.strip().upper()
        if up in choices:
            return up
        for k, v in choices.items():
            if isinstance(v, str) and v.strip() == ans.strip():
                return k
        if up.isdigit():
            idx = int(up)
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            if 1 <= idx <= len(choices):
                return letters[idx - 1]
            if 0 <= idx < len(choices):
                return letters[idx]
        return "A" if choices else ""
    if isinstance(ans, int):
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if 1 <= ans <= len(choices):
            return letters[ans - 1]
        if 0 <= ans < len(choices):
            return letters[ans]
        return "A" if choices else ""
    return "A" if choices else ""

# --- text extraction ---

def extract_question_text(obj: dict) -> str:
    return (
        obj.get("question")
        or obj.get("question_ko")
        or obj.get("prompt")
        or obj.get("prompt_ko")
        or ""
    )

def extract_passage(obj: dict) -> str:
    for k in ("passage", "passage_ko"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    for k in ("passage", "passage_ko"):
        v = obj.get(k)
        if isinstance(v, list):
            items: List[str] = []
            for it in v:
                if isinstance(it, str):
                    s = it.strip()
                    if s:
                        items.append(s)
                elif isinstance(it, dict):
                    t = (it.get("text") or it.get("utterance") or "").strip()
                    if t:
                        items.append(t)
            if items:
                return "\n".join(items)
    return ""

def extract_dialog(obj: dict) -> List[dict]:
    dlg = obj.get("dialogue")
    # list of dict/str
    if isinstance(dlg, list) and dlg:
        out: List[dict] = []
        for turn in dlg:
            if isinstance(turn, dict):
                spk = turn.get("speaker") or turn.get("role") or "지문"
                txt = turn.get("text") or turn.get("utterance") or ""
                if txt:
                    out.append({"speaker": spk, "text": txt})
            elif isinstance(turn, str):
                t = turn.strip()
                if t:
                    out.append({"speaker": "지문", "text": t})
        if out:
            return out
    # dialogue dict
    if isinstance(dlg, dict):
        inner = dlg.get("turns")
        if isinstance(inner, list) and inner:
            out: List[dict] = []
            for turn in inner:
                if isinstance(turn, dict):
                    spk = turn.get("speaker") or turn.get("role") or "지문"
                    txt = turn.get("text") or turn.get("utterance") or ""
                    if txt:
                        out.append({"speaker": spk, "text": txt})
                elif isinstance(turn, str):
                    t = turn.strip()
                    if t:
                        out.append({"speaker": "지문", "text": t})
            if out:
                return out
        spk = dlg.get("speaker") or dlg.get("role") or "지문"
        txt = dlg.get("text") or dlg.get("utterance") or ""
        if isinstance(txt, str) and txt.strip():
            return [{"speaker": spk, "text": txt.strip()}]
    # dialogue string
    if isinstance(dlg, str):
        t = dlg.strip()
        if t:
            return [{"speaker": "지문", "text": t}]
    # fallback: passage_ko
    passage = obj.get("passage_ko")
    if passage is None:
        return []
    out: List[dict] = []
    if isinstance(passage, str):
        lines = [ln.strip() for ln in re.split(r"[\r\n]+", passage) if ln.strip()]
        for ln in lines:
            m = _SPEAKER_RE.match(ln)
            if m:
                spk, txt = m.group(1), m.group(2).strip()
                out.append({"speaker": spk, "text": txt})
            else:
                out.append({"speaker": "지문", "text": ln})
        return out
    if isinstance(passage, list):
        for item in passage:
            if isinstance(item, dict):
                spk = item.get("speaker") or item.get("role") or "지문"
                txt = item.get("text") or item.get("utterance") or ""
                if txt:
                    out.append({"speaker": spk, "text": txt})
            elif isinstance(item, str):
                m = _SPEAKER_RE.match(item.strip())
                if m:
                    spk, txt = m.group(1), m.group(2).strip()
                    out.append({"speaker": spk, "text": txt})
                else:
                    out.append({"speaker": "지문", "text": item.strip()})
        return out
    return []