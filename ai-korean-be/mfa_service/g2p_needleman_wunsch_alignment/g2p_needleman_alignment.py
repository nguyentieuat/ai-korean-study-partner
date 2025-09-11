# g2p_needleman_alignment.py
import os
import json
import requests
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from textgrid import TextGrid

from utils.utils import load_mfa_dict, text_to_phonemes_mfa
from .scoring import score_with_textgrid, score_chunks_vs_hyp

ASR_SERVICE_URL = os.getenv("ASR_SERVICE_URL", "http://127.0.0.1:5005")
ASR_ENDPOINT = f"{ASR_SERVICE_URL}/api/stt"
MFA_DICT_PATH = "./korean_mfa.dict"

def evaluate(
    audio_file: str,
    reference_text: str,
    textgrid_file: Optional[str] = None,
) -> Dict[str, Any]:
    """ASR -> G2P (MFA) -> alignment (base) -> SURFACE scoring + advice (handles liaison/assimilation)."""
    # 1) ASR -> user transcript
    files = {}
    if audio_file and os.path.exists(audio_file):
        files["file"] = (os.path.basename(audio_file), open(audio_file, "rb"), "audio/wav")
    try:
        if files:
            r = requests.post(ASR_ENDPOINT, files=files, timeout=60)
            r.raise_for_status()
            asr_json = r.json()
            user_transcript = asr_json.get("text", "") or asr_json.get("transcript", "") or ""
        else:
            user_transcript = ""
    except Exception:
        user_transcript = ""

    # 2) G2P with MFA dict (BASE sequences)
    try:
        mfa_dict = load_mfa_dict(MFA_DICT_PATH)
    except Exception:
        mfa_dict = {}
    ref_chunks, notes = text_to_phonemes_mfa(reference_text, mfa_dict, return_by_word=True) or []
    hyp_phonemes = text_to_phonemes_mfa(user_transcript, mfa_dict) or []

    # 3) Tính điểm
    if textgrid_file:
        # DÙNG phones & duration từ TextGrid
        res = score_with_textgrid(
            textgrid_file,
            hyp_phonemes,
            ref_chunks=ref_chunks,          # ← quan trọng: truyền theo tên
            word_tier="words",
            phone_tier="phones",
            collapse_diphthongs=True,
            duration_weight="linear",
            gap_penalty=-0.2,
            mismatch_penalty=-0.5,
            drop_phone_labels={"sil","sp","pau","","spn","nsn","brth","noise","laugh","tsk"},
        )
    else:
        # Fallback: G2P ref_chunks (không có duration)
        res = score_chunks_vs_hyp(
            ref_chunks, hyp_phonemes,
            gap_penalty=-0.2, mismatch_penalty=-0.5,
            advice_threshold=0.85,
            collapse_diphthongs=True
        )
    return {
        "note": notes,
        **res
    }

if __name__ == "__main__":
    # quick smoke (requires reachable ASR + optional TextGrid path)
    import sys
    audio = sys.argv[1] if len(sys.argv) > 1 else ""
    ref = sys.argv[2] if len(sys.argv) > 2 else "정말 미안해요"
    tg = sys.argv[3] if len(sys.argv) > 3 else None
    out = evaluate(audio_file=audio, reference_text=ref, textgrid_file=tg)
    print(json.dumps(out, ensure_ascii=False, indent=2))
