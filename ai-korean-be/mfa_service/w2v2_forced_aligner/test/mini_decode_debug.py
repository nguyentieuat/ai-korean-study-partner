#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mini debug: chạy gate + decode + confidences cho 1 file wav và 1 câu tham chiếu (NFD Jamo).
- In ra: vocab_mode, hyp decode, CER, NLL/char, gate flags, mean conf, v.v.
- Hữu ích để kiểm tra “decode không ra” do mismatch vocab/VAD.
- Yêu cầu: đã cài torchaudio, transformers và 2 file cốt lõi:
    - w2v2_forced_scoring.py
    - ctc_gate.py  (trong package w2v2_forced_aligner hoặc sửa import bên dưới)
"""

import os
import argparse
import torch
import torchaudio
import unicodedata as ud
from typing import Optional, Tuple

# === IMPORT: chỉnh lại path/module nếu bạn đặt file khác vị trí
from w2v2_forced_aligner.w2v2_forced_scoring import score_pronunciation_forced
from w2v2_forced_aligner.ctc_gate import (
    ctc_gate_global_from_waveform,
    ctc_char_confidences_from_waveform,
)

def to_nfd(s: str) -> str:
    return ud.normalize("NFKD", s)

def load_audio(wav_path: str) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(wav_path)
    if wav.ndim == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav, sr
def is_jamo(ch: str) -> bool:
    cp = ord(ch) if ch else 0
    return 0x1100 <= cp <= 0x11FF  # chỉ giữ Jamo NFD

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="Đường dẫn wav 16kHz mono (audio gốc của bạn).")
    ap.add_argument("--ref", required=True, help="Chuỗi tham chiếu (Hangul), sẽ chuẩn hoá về NFD Jamo.")
    ap.add_argument("--chunks", default="", help="Chuỗi syllable tách bằng '/', ví dụ: '안/녕'. Optional.")
    ap.add_argument("--print-syllables", action="store_true", help="In per-syllable jamo confidences.")
    ap.add_argument("--use-textgrid", default="", help="Đường dẫn TextGrid (optional) để làm duration priors.")
    args = ap.parse_args()

    wav, sr = load_audio(args.wav)
    ref = to_nfd(args.ref)

    # Tạo ref_chunks nếu người dùng muốn (để hiển thị jamo per-syllable)
    ref_chunks = None
    if args.chunks.strip():
        raw_chunks = [c for c in args.chunks.split("/") if c]
        ref_chunks = []
        for syl in raw_chunks:
            # NFD rồi lọc chỉ còn jamo để chấm điểm (loại space, dấu câu, v.v.)
            jamos_all = list(to_nfd(syl))
            jamos_only = [ch for ch in jamos_all if is_jamo(ch)]
            if not jamos_only:
                # chunk toàn space/dấu câu -> bỏ khỏi scoring để tránh len mismatch
                continue
            # Lưu ý: 'syl' vẫn giữ nguyên cho mục đích hiển thị
            ref_chunks.append((syl, jamos_only))

        # Nếu lỡ tay truyền cả câu thành 1 chunk (rất dài), cảnh báo nhẹ
        if len(ref_chunks) == 1 and len(ref_chunks[0][1]) > 10:
            print("[WARN] You passed one very long chunk. Consider splitting by syllables for better per-syllable display.")

    print("=== MINI DECODE DEBUG ===")
    print(f"WAV: {args.wav}")
    print(f"REF (raw): {args.ref}")
    print(f"REF (NFD): {ref}")
    if ref_chunks:
        lens = [len(j) for _, j in ref_chunks]
        print(f"REF chunks: {len(ref_chunks)} syllables; jamo per syllable = {lens}")

    # 1) Gate (lai): in đầy đủ chỉ số
    g = ctc_gate_global_from_waveform(
        wav, sr, ref_text_nfd=ref,
        cer_threshold=float(os.getenv("W2V_CTC_CER_THRESHOLD", "0.65")),
        loss_char_threshold=float(os.getenv("W2V_CTC_LOSS_CHAR_THRESHOLD", "2.0")),
        ref_nll_threshold=float(os.getenv("W2V_CTC_REF_NLL_THR", "2.2")),
    )
    print("\n[Gate]")
    for k in ["cer","cer_skeleton","mean_neglogp","ref_nll_per_char","gate_by_pair","gate_by_nll","gate_by_short","gate_pass"]:
        print(f"- {k}: {g.get(k)}")

    # === HIỂN THỊ: giữ nguyên space/dấu câu (các field đã được bổ sung trong ctc_gate.py) ===
    # hyp_display: chuỗi decode thô (có separators), hyp_display_nfd: NFD giữ separators,
    # hyp_nfd: NFD dùng để tính CER (đã loại bỏ separators),
    # ref_display_nfd: NFD của ref để hiển thị (giữ separators)
    print("\n[Text display]")
    print(f"- hyp (raw)                  : {g.get('hyp_display')}")
    print(f"- hyp (NFD display, keep SP) : {g.get('hyp_display_nfd')}")
    print(f"- hyp (NFD scoring)          : {g.get('hyp_nfd')}")
    print(f"- ref (NFD display, keep SP) : {g.get('ref_display_nfd')}")

    # 2) Confidences thô (per-jamo hoặc per-syllable tuỳ vocab mode)
    confs = ctc_char_confidences_from_waveform(
        wav, sr, ref_text_nfd=ref,
        blank_vad_thresh=float(os.getenv("W2V_BLANK_VAD_THRESH", "0.80")),
        vad_pad_ms=float(os.getenv("W2V_VAD_PAD_MS", "280.0")),
        temp=float(os.getenv("W2V_LOGIT_TEMP", "1.6")),
        use_confusables=True,
        priors_sec=None  # set từ TextGrid nếu muốn
    )
    print("\n[Confidences]")
    if len(confs) <= 64:
        print("confs:", [round(float(c),4) for c in confs])
    else:
        print(f"confs: len={len(confs)}; mean={sum(confs)/max(1,len(confs)):.4f}")

    # 3) Scoring tổng hợp + breakdown (đã auto-pad 0.2s cho clip ngắn)
    out = score_pronunciation_forced(
        waveform=wav, waveform_sr=sr,
        reference_text=ref, ref_chunks=ref_chunks,
        ref_textgrid_path=(args.use_textgrid or None),
        tau_char_power=float(os.getenv("W2V_TAU", "1.0")),
        lam_edit=float(os.getenv("W2V_LAM_EDIT", "0.0")),
    )
    print("\n[Final Score]")
    print("score:", out.get("score"))
    if not out.get("gate_pass", True):
        print("score_soft (ref):", out.get("score_soft"))

    details = out.get("details", {})
    print("\n[Details summary]")
    print({k: details.get(k) for k in ["missing_ratio","len_mismatch","lens","params","score_formula"] if k in details})

    if args.print_syllables:
        print("\n[Per-syllable]")
        for i, syl in enumerate(out.get("details_collapsed", []), 1):
            jamos = syl.get("jamo", [])
            confs = [round(float(j.get("conf",0.0)),3) for j in jamos]
            marks = ["(S)" if j.get("silent") else "" for j in jamos]
            print(f"{i:02d}. {syl.get('text','')} [{syl.get('text_nfd','')}] conf={round(float(syl.get('conf',0.0)),3)} -> {list(zip([j.get('jamo','') for j in jamos], confs, marks))}")

if __name__ == "__main__":
    main()
