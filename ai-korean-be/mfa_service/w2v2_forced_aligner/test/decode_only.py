#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Decode-only (best-path) + optional: per-unit confidences + gate check.
- --print-conf: in confidences per jamo/syllable (CTC Viterbi)
- --check-gate : gọi ctc_gate_global_from_waveform và in các chỉ số gate
"""

import os
import argparse
import unicodedata as ud
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Nếu chạy file trực tiếp và bị lỗi import package, bỏ comment 4 dòng dưới:
# import sys
# from pathlib import Path
# PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]  # thư mục chứa package w2v2_forced_aligner
# sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from ctc_gate import (
    ctc_char_confidences_from_waveform,
    ctc_gate_global_from_waveform,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR = 16000

def to_nfd_jamo_only(s: str) -> str:
    s = ud.normalize("NFKD", s)
    return "".join(ch for ch in s if 0x1100 <= ord(ch) <= 0x11FF)

def vocab_mode(vocab: dict) -> str:
    has_nfd = ("ᄋ" in vocab) or ("ᄀ" in vocab)
    has_compat = ("ㅇ" in vocab) or ("ㄱ" in vocab) or ("ㅏ" in vocab)
    if has_nfd: return "nfd_jamo"
    if has_compat: return "compat_jamo"
    return "syllable"

def blank_vad(logprobs: torch.Tensor, blank_id: int, thr: float = 0.80, pad_ms: float = 250.0) -> slice:
    with torch.no_grad():
        blank_post = logprobs.exp()[:, blank_id]
        nz = (blank_post < thr).nonzero(as_tuple=False).squeeze(-1)
        if nz.numel() == 0:
            return slice(0, logprobs.shape[0])
        start = nz[0].item()
        end = nz[-1].item() + 1
        pad_frames = int(round(pad_ms / 20.0))  # ~20ms/frame
        start = max(0, start - pad_frames)
        end = min(logprobs.shape[0], end + pad_frames)
        return slice(start, end)

def best_path_decode(logprobs: torch.Tensor, id2tok: dict, blank_id: int) -> str:
    ids = logprobs.argmax(dim=-1).tolist()
    out, prev = [], None
    for i in ids:
        if i == blank_id:
            prev = i; continue
        if i != prev:
            out.append(id2tok.get(i, ""))
        prev = i
    return "".join(out)

def cer(a: str, b: str) -> float:
    if not b:
        return 1.0 if a else 0.0
    if a == b:
        return 0.0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j-1] + 1
            dele = prev[j] + 1
            rep = prev[j-1] + (ca != cb)
            cur.append(min(ins, dele, rep))
        prev = cur
    return prev[-1] / max(1, len(b))

def load_audio(path: str):
    wav, sr = torchaudio.load(path)
    if wav.ndim == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
        sr = TARGET_SR
    return wav, sr

def run_gate_check(wav: torch.Tensor, sr: int, ref_hangul: str,
                   cer_thr: float, loss_char_thr: float, ref_nll_thr: float,
                   vad_thr: float, vad_pad_ms: float, temp: float) -> None:
    """Gọi ctc_gate_global_from_waveform và in kết quả gọn gàng."""
    ref_nfd = ud.normalize("NFKD", ref_hangul) if ref_hangul else ""
    g = ctc_gate_global_from_waveform(
        wav, sr, ref_text_nfd=ref_nfd,
        cer_threshold=cer_thr,
        loss_char_threshold=loss_char_thr,
        ref_nll_threshold=ref_nll_thr,
        short_syll_cer_bonus=0.05,
    )
    print("\n=== GATE CHECK ===")
    # các key in theo thứ tự dễ đọc
    keys = ["gate_pass","gate_by_short","gate_by_pair","gate_by_nll",
            "cer","cer_skeleton","mean_neglogp","ref_nll_per_char"]
    for k in keys:
        if k in g:
            v = g[k]
            print(f"{k:>16}: {v}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="Đường dẫn audio (wav/mp3…).")
    ap.add_argument("--ref", default="", help="Chuỗi tham chiếu (Hangul). Optional.")
    ap.add_argument("--model", default=os.getenv("W2V_MODEL", "Kkonjeong/wav2vec2-base-korean"),
                    help="Tên/đường dẫn checkpoint Wav2Vec2-CTC.")
    ap.add_argument("--temp", type=float, default=float(os.getenv("W2V_LOGIT_TEMP", "1.5")), help="Logit temperature.")
    ap.add_argument("--vad-thr", type=float, default=float(os.getenv("W2V_BLANK_VAD_THRESH", "0.80")), help="Ngưỡng blank posterior cho VAD.")
    ap.add_argument("--vad-pad-ms", type=float, default=float(os.getenv("W2V_VAD_PAD_MS", "250.0")), help="Pad ms mỗi đầu vùng thoại.")
    ap.add_argument("--print-conf", action="store_true", help="In per-unit confidences (CTC Viterbi).")
    ap.add_argument("--check-gate", action="store_true", help="Kiểm tra ctc_gate_global_from_waveform và in các chỉ số.")
    # ngưỡng gate (override nếu muốn)
    ap.add_argument("--cer-thr", type=float, default=float(os.getenv("W2V_CTC_CER_THRESHOLD", "0.65")))
    ap.add_argument("--loss-char-thr", type=float, default=float(os.getenv("W2V_CTC_LOSS_CHAR_THRESHOLD", "2.0")))
    ap.add_argument("--ref-nll-thr", type=float, default=float(os.getenv("W2V_CTC_REF_NLL_THR", "2.2")))
    args = ap.parse_args()

    # 1) Load model/tokenizer
    print(f"[Info] Loading model: {args.model}")
    proc = Wav2Vec2Processor.from_pretrained(args.model)
    model = Wav2Vec2ForCTC.from_pretrained(args.model).to(DEVICE).eval()
    vocab = proc.tokenizer.get_vocab()
    id2tok = {v:k for k,v in vocab.items()}
    blank_id = int(getattr(proc.tokenizer, "pad_token_id", vocab.get("<pad>", 0)))
    mode = vocab_mode(vocab)
    print(f"[Info] vocab_mode={mode}, blank_id={blank_id}, vocab_size={len(vocab)}")

    # 2) Audio -> logits -> logprobs
    wav, sr = load_audio(args.wav)
    with torch.inference_mode():
        inputs = proc(wav.squeeze(0), sampling_rate=TARGET_SR, return_tensors="pt", padding=False)
        logits = model(inputs.input_values.to(DEVICE)).logits[0]  # [T,V]
        logprobs_full = torch.log_softmax(logits / max(1e-6, args.temp), dim=-1)

        # 3) VAD crop + fallback
        sl = blank_vad(logprobs_full, blank_id, thr=args.vad_thr, pad_ms=args.vad_pad_ms)
        logprobs = logprobs_full[sl]
        if logprobs.shape[0] < 6:  # fallback nếu crop quá ngắn
            logprobs = logprobs_full
            sl = slice(0, logprobs_full.shape[0])

    # 4) Decode
    hyp = best_path_decode(logprobs, id2tok, blank_id)
    hyp_nfd = to_nfd_jamo_only(hyp)
    print("\n=== DECODE ===")
    print(f"Hyp (vocab tokens): {hyp}")
    print(f"Hyp_NFD (Jamo):    {hyp_nfd}")
    print(f"T_full={logprobs_full.shape[0]}  T_crop={logprobs.shape[0]}")

    # 5) CER nếu có tham chiếu
    if args.ref:
        ref_nfd = to_nfd_jamo_only(ud.normalize("NFKD", args.ref))
        c = cer(hyp_nfd, ref_nfd)
        print(f"REF: {args.ref}  | REF_NFD: {ref_nfd}")
        print(f"CER (Jamo-NFD): {c:.3f}")

    # 6) (Optional) per-unit confidences
    if args.print_conf:
        confs = ctc_char_confidences_from_waveform(
            wav, sr, ref_text_nfd=ud.normalize("NFKD", args.ref or hyp),
            blank_vad_thresh=args.vad_thr, vad_pad_ms=args.vad_pad_ms, temp=args.temp,
            use_confusables=True, priors_sec=None
        )
        if len(confs) <= 64:
            print(f"Confidences ({len(confs)}):", [round(float(c), 3) for c in confs])
        else:
            print(f"Confidences: len={len(confs)}, mean={sum(confs)/max(1,len(confs)):.3f}")

    # 7) (Optional) Gate check
    if args.check_gate and args.ref:
        run_gate_check(
            wav, sr, args.ref,
            cer_thr=args.cer_thr, loss_char_thr=args.loss_char_thr, ref_nll_thr=args.ref_nll_thr,
            vad_thr=args.vad_thr, vad_pad_ms=args.vad_pad_ms, temp=args.temp
        )
    elif args.check_gate and not args.ref:
        print("\n[Warn] --check-gate yêu cầu --ref để tính CER / NLL theo tham chiếu.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if "sox" in str(e).lower() or "mp3" in str(e).lower():
            print("[Hint] Nếu lỗi đọc MP3: cài backend (pip install soundfile) hoặc convert sang WAV 16kHz.")
        raise
