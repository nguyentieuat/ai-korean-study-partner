# pytest -q tests/test_ctc_decode_mini.py
# Thiết lập:
#   export WAV1=/path/to/annyeong.wav
#   export REF1=안녕
#   (tuỳ chọn) export CHUNKS1="안/녕"
#
# Test sẽ:
#  - Chạy gate và assert rằng code không crash
#  - In ra hyp/gate/score (không assert giá trị tuyệt đối vì phụ thuộc checkpoint)
#  - Nếu không có WAV1/REF1 thì skip

import os
import pytest
import torchaudio
import unicodedata as ud

from w2v2_forced_aligner.w2v2_forced_scoring import score_pronunciation_forced
from w2v2_forced_aligner.ctc_gate import (
    ctc_gate_global_from_waveform,
    ctc_char_confidences_from_waveform,
)

WAV1 = os.getenv("WAV1", "data/first/audio/reply_642207af8d664e05a709d51b953b8ea7.mp3")
REF1 = os.getenv("REF1", "가")
CHUNKS1 = os.getenv("CHUNKS1", "가")

def to_nfd(s: str) -> str:
    return ud.normalize("NFKD", s)

@pytest.mark.skipif(not WAV1 or not os.path.exists(WAV1) or not REF1, reason="Cần WAV1 & REF1 env để chạy test")
def test_decode_gate_confidences_end2end(capfd):
    wav, sr = torchaudio.load(WAV1)
    ref = to_nfd(REF1)

    ref_chunks = None
    if CHUNKS1.strip():
        parts = [p for p in CHUNKS1.split("/") if p]
        ref_chunks = [(p, list(to_nfd(p))) for p in parts]

    # Gate — chỉ cần không crash và có trả về dict có các keys quan trọng
    g = ctc_gate_global_from_waveform(wav, sr, ref_text_nfd=ref)
    for k in ["cer","cer_skeleton","mean_neglogp","ref_nll_per_char","gate_pass"]:
        assert k in g

    # Confidences — vector float
    confs = ctc_char_confidences_from_waveform(wav, sr, ref_text_nfd=ref)
    assert isinstance(confs, list)
    assert all(isinstance(c, float) for c in confs)

    # Score tổng — có trường "score"
    out = score_pronunciation_forced(
        waveform=wav, waveform_sr=sr, reference_text=ref, ref_chunks=ref_chunks
    )
    assert "score" in out
    # In một số thông tin debug ra stdout (để inspect khi cần)
    print("gate:", g)
    print("score:", out.get("score"))
    if not out.get("gate_pass", True):
        print("score_soft:", out.get("score_soft"))
