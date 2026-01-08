#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import time
import uuid
import math
import logging
from typing import Optional, Deque
from collections import deque

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

# -------------------- Config --------------------
SAMPLE_RATE_DEFAULT = int(os.getenv("ASR_SAMPLE_RATE", "16000"))
FINAL_SILENCE_MS    = int(os.getenv("ASR_FINAL_SILENCE_MS", "700"))
MAX_SEGMENT_SEC     = float(os.getenv("ASR_MAX_SEGMENT_SEC", "12.0"))
CHUNK_WARN_BYTES    = int(os.getenv("ASR_CHUNK_WARN_BYTES", "4096"))
USE_WEBRTCVAD       = os.getenv("ASR_USE_WEBRTCVAD", "0") == "1"
VAD_AGGRESSIVENESS  = int(os.getenv("ASR_VAD_AGGR", "2"))
LOG_LEVEL           = os.getenv("LOG_LEVEL", "INFO")

# Rolling-window decode (server-side)
ROLLING_WINDOW_SEC  = float(os.getenv("ASR_ROLLING_WINDOW_SEC", "1.3"))   # 1.2–1.5s
ROLLING_EMIT_MIN_GAIN = int(os.getenv("ASR_ROLLING_EMIT_MIN_GAIN", "2"))  # tăng ≥2 ký tự mới thì emit
ROLLING_EMIT_MAX_LAG_MS = int(os.getenv("ASR_ROLLING_EMIT_MAX_LAG_MS", "900"))  # ép emit nếu quá lâu
ROLLING_MIN_HYP_LEN = int(os.getenv("ASR_ROLLING_MIN_HYP_LEN", "3"))      # bỏ hyp quá ngắn

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("asr-ws")

# ==== ASR Runtime (HF Wav2Vec2 + CTC) =======================================
import torch, torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

ASR_MODEL_NAME = os.getenv("W2V_MODEL", "Kkonjeong/wav2vec2-base-korean")
ASR_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ASR_FP16 = os.getenv("ASR_FP16", "1") == "1" and ASR_DEVICE == "cuda"

log.info("Loading ASR model: %s on %s", ASR_MODEL_NAME, ASR_DEVICE)
processor = Wav2Vec2Processor.from_pretrained(ASR_MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(ASR_MODEL_NAME).to(ASR_DEVICE).eval()
if ASR_FP16:
    model = model.half()

def _pcm16_to_tensor(pcm16: bytes, sr: int, target_sr: int = SAMPLE_RATE_DEFAULT) -> torch.Tensor:
    wav = torch.frombuffer(pcm16, dtype=torch.int16).to(torch.float32) / 32768.0  # [-1,1]
    wav = wav.unsqueeze(0)  # [1, T]
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.to(ASR_DEVICE, non_blocking=True)

@torch.no_grad()
def _ctc_greedy_decode(wav_1xT: torch.Tensor) -> tuple[str, float]:
    inputs = processor(wav_1xT.cpu().numpy(), sampling_rate=SAMPLE_RATE_DEFAULT, return_tensors="pt", padding=False)
    input_values = inputs.input_values.to(ASR_DEVICE)
    logits = model(input_values).logits                     # [1, T, V]
    probs  = torch.softmax(logits, dim=-1)                  # [1, T, V]
    maxp   = probs.max(dim=-1).values.squeeze(0)            # [T]
    ids    = torch.argmax(logits, dim=-1).cpu().numpy()     # [1, T]
    text   = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    conf   = float(maxp[maxp > 0.1].mean().item()) if (maxp > 0.1).any() else float(maxp.mean().item())
    return text, conf

# Optional ctc_gate for better confidence on final segments
USE_CTC_GATE = os.getenv("USE_CTC_GATE", "0") == "1"
if USE_CTC_GATE:
    try:
        from core.ctc_gate import ctc_char_confidences_from_waveform
        log.info("ctc_gate confidence enabled")
    except Exception as e:
        log.warning("ctc_gate import failed: %s", e)
        USE_CTC_GATE = False

# ---------------- FastAPI Router ----------------
router = APIRouter(prefix="", tags=["asr_ws"])

# Optional: webrtcvad
_vad = None
if USE_WEBRTCVAD:
    try:
        import webrtcvad  # type: ignore
        _vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        log.info("Using webrtcvad (aggr=%d)", VAD_AGGRESSIVENESS)
    except Exception as e:
        log.warning("Failed to import webrtcvad: %s ; fallback to RMS VAD.", e)
        _vad = None

# -------------------- Utilities --------------------
def now_ms() -> int:
    return int(time.time() * 1000)

def pcm16_bytes_to_np(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.int16)

def frame_ms_to_samples(ms: int, sr: int) -> int:
    return int(sr * (ms / 1000.0))

def rms_dbfs(x: np.ndarray) -> float:
    if x.size == 0:
        return -100.0
    xf = x.astype(np.float32) / 32768.0
    rms = np.sqrt(np.mean(xf * xf) + 1e-12)
    return 20.0 * math.log10(rms + 1e-12)

def vad_is_speech(chunk: bytes, sr: int) -> bool:
    arr = pcm16_bytes_to_np(chunk)
    if arr.size == 0:
        return False
    if _vad is not None:
        frame_dur_ms = 20
        frame_len = frame_ms_to_samples(frame_dur_ms, sr)
        arr_bytes = arr.tobytes()
        speech_votes = 0
        total = 0
        for off in range(0, len(arr_bytes), frame_len * 2):
            sub = arr_bytes[off : off + frame_len * 2]
            if len(sub) < frame_len * 2:
                break
            ok = _vad.is_speech(sub, sr)
            speech_votes += 1 if ok else 0
            total += 1
        if total == 0:
            return False
        return (speech_votes / total) >= 0.5
    db = rms_dbfs(arr)
    return db > -40.0

# -------------------- Hangul compose (compat jamo -> syllables) -------------
_COMP_CHO  = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_COMP_JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_COMP_JONG = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")
_CHO_MAP  = {ch: i for i, ch in enumerate(_COMP_CHO)}
_JUNG_MAP = {ch: i for i, ch in enumerate(_COMP_JUNG)}
_JONG_MAP = {ch: i for i, ch in enumerate(_COMP_JONG) if ch}

def compose_compat_jamo_to_hangul(s: str) -> str:
    """Ghép chuỗi compat-jamo (CV(C?)) thành âm tiết Hangul NFC."""
    out = []
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch in _CHO_MAP:
            cho = _CHO_MAP[ch]; i += 1
        else:
            out.append(ch); i += 1; continue
        if i >= n or s[i] not in _JUNG_MAP:
            out.append(_COMP_CHO[cho])  # onset rời
            continue
        jung = _JUNG_MAP[s[i]]; i += 1
        jong = 0
        if i < n and s[i] in _JONG_MAP:
            if not (i+1 < n and s[i+1] in _JUNG_MAP):  # nếu sau là vowel thì coi như onset của âm tiết sau
                jong = _JONG_MAP[s[i]]; i += 1
        syll = 0xAC00 + (cho * 21 + jung) * 28 + jong
        out.append(chr(syll))
    return "".join(out)

# -------------------- Merge helper (suffix–prefix) --------------------------
import unicodedata as _ud

def _norm(s: str) -> str:
    if not s:
        return ""
    def is_sep(ch):
        cat = _ud.category(ch)
        return cat.startswith('Z') or cat.startswith('P')
    return "".join(ch for ch in _ud.normalize("NFKD", s) if not is_sep(ch))

def merge_with_overlap(prev: str, curr: str, min_overlap: int = 4) -> str:
    """Gộp chuỗi với chồng lặp suffix–prefix (so sánh trên _norm)."""
    if not prev:
        return curr
    if not curr:
        return prev
    pn, cn = _norm(prev), _norm(curr)
    max_ol = min(len(pn), len(cn))
    best = 0
    for L in range(max_ol, min_overlap - 1, -1):
        if pn.endswith(cn[:L]):
            best = L
            break
    if best <= 0:
        joiner = "" if (prev.endswith(" ") or curr.startswith(" ")) else " "
        return prev + joiner + curr
    cut = 0
    while cut < len(curr) and len(_norm(curr[:cut])) < best:
        cut += 1
    return prev + curr[cut:]

# -------------------- ASR: rolling-window partial ---------------------------
async def asr_partial_decode_rolling(rolling_bytes: bytes, sr: int) -> str:
    """Decode toàn bộ cửa sổ trượt rồi compose Hangul trước khi trả về."""
    if not rolling_bytes:
        return ""
    wav = _pcm16_to_tensor(rolling_bytes, sr)
    raw, _ = _ctc_greedy_decode(wav)
    if not raw:
        return ""
    compat_ratio = sum(ch in _CHO_MAP or ch in _JUNG_MAP or ch in _JONG_MAP for ch in raw) / max(1, len(raw))
    if compat_ratio >= 0.3:
        try:
            raw = compose_compat_jamo_to_hangul(raw)
        except Exception:
            pass
    return " ".join(raw.split())

async def asr_final_decode(pcm16: bytes, sr: int) -> dict:
    """Decode toàn segment khi im lặng dài/EOS. Trả về text + conf + thời lượng."""
    if not pcm16:
        return {"text": "", "conf": 0.0, "words": [], "t0": 0.0, "t1": 0.0}
    wav = _pcm16_to_tensor(pcm16, sr)
    text, conf = _ctc_greedy_decode(wav)
    compat_ratio = sum(ch in _CHO_MAP or ch in _JUNG_MAP or ch in _JONG_MAP for ch in text) / max(1, len(text))
    if compat_ratio >= 0.3:
        try:
            text = compose_compat_jamo_to_hangul(text)
        except Exception:
            pass
    if USE_CTC_GATE:
        try:
            confs = ctc_char_confidences_from_waveform(wav.detach().cpu(), SAMPLE_RATE_DEFAULT, text)
            if confs:
                conf = float(sum(confs) / max(1, len(confs)))
        except Exception as e:
            log.debug("ctc_gate confidence failed: %s; fallback avg prob", e)
    dur = len(pcm16) / (2 * sr)
    return {"text": text, "conf": round(conf, 4), "words": [], "t0": 0.0, "t1": round(dur, 2)}

def compute_metrics(seg_bytes: bytes, sr: int) -> dict:
    arr = pcm16_bytes_to_np(seg_bytes)
    db = rms_dbfs(arr); dur = len(arr) / sr
    return {"rms_dbfs": round(db, 2), "dur_sec": round(dur, 2), "sample_rate": sr}

# -------------------- Session --------------------
class Session:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.id = uuid.uuid4().hex[:8]
        self.sr = SAMPLE_RATE_DEFAULT
        self.lang = "ko"
        self.started_ms = now_ms()

        # buffer dồn toàn segment (để VAD & final decode)
        self.seg_buf = bytearray()
        self.last_voice_ms = now_ms()

        # rolling window buffer (giới hạn ~1.3s)
        self.roll_buf = bytearray()
        self.roll_cap_bytes = int(2 * self.sr * ROLLING_WINDOW_SEC)  # 2 bytes/sample

        # merge state
        self.partial_text = ""
        self.last_emit_ms = 0

        # meta (tùy chọn)
        self.pending_meta: Optional[dict] = None
        self.seq_expected: int = 0

        # de-overlap state
        self.seen_first_chunk = False
        self.next_overlap_bytes = 0
        self.next_chunk_ms = 0
        self.next_overlap_ms = 0

        self.chunk_hist: Deque[int] = deque(maxlen=16)

    async def sendj(self, obj: dict):
        if self.ws.application_state == WebSocketState.CONNECTED:
            await self.ws.send_text(json.dumps(obj, ensure_ascii=False))

    def note_chunk(self, nbytes: int):
        self.chunk_hist.append(nbytes)

    def append_to_rolling(self, chunk: bytes):
        """Nạp chunk vào cửa sổ trượt và cắt bớt phần quá cap."""
        self.roll_buf.extend(chunk)
        if len(self.roll_buf) > self.roll_cap_bytes:
            overflow = len(self.roll_buf) - self.roll_cap_bytes
            del self.roll_buf[:overflow]

# -------------------- WebSocket Endpoint --------------------
@router.websocket("/ws/asr")
async def ws_asr(ws: WebSocket):
    await ws.accept()
    sess = Session(ws)
    await sess.sendj({"event": "ready", "session_id": sess.id})

    try:
        while True:
            msg = await ws.receive()

            # ----- text frames -----
            if "text" in msg:
                try:
                    data = json.loads(msg["text"])
                except Exception:
                    await sess.sendj({"event": "error", "detail": "Invalid JSON"})
                    continue

                ev = data.get("event")

                if ev == "init":
                    sess.sr = int(data.get("sample_rate") or SAMPLE_RATE_DEFAULT)
                    sess.lang = data.get("lang") or "ko"
                    fmt = data.get("format") or "pcm_s16le"
                    if fmt != "pcm_s16le":
                        await sess.sendj({"event": "error", "detail": "Only pcm_s16le is supported"})
                        continue
                    sess.roll_cap_bytes = int(2 * sess.sr * ROLLING_WINDOW_SEC)
                    await sess.sendj({"event": "ack", "sr": sess.sr})

                elif ev == "chunk_meta":
                    # meta cho CHUNK KẾ TIẾP
                    try:
                        cm = int(data.get("chunk_ms", 300))
                        om = int(data.get("overlap_ms", 0))
                        sess.pending_meta = {
                            "seq": int(data["seq"]),
                            "chunk_ms": cm,
                            "overlap_ms": om,
                        }
                        sess.next_chunk_ms = cm
                        sess.next_overlap_ms = om
                        sess.next_overlap_bytes = int(sess.sr * (om / 1000.0)) * 2  # int16 mono
                    except Exception:
                        sess.pending_meta = None
                        sess.next_overlap_bytes = 0

                elif ev == "eos":
                    # finalize remaining buffer if any
                    if len(sess.seg_buf) > 0:
                        res = await asr_final_decode(bytes(sess.seg_buf), sess.sr)
                        await sess.sendj({"event": "final", **res})
                        metrics = compute_metrics(sess.seg_buf, sess.sr)
                        await sess.sendj({"event": "metrics", **metrics})
                        sess.seg_buf.clear()
                    await sess.sendj({"event": "done"})
                    break

                else:
                    pass

            # ----- binary frames (audio) -----
            elif "bytes" in msg:
                raw: bytes = msg["bytes"]
                nbytes = len(raw)
                if nbytes == 0:
                    continue
                sess.note_chunk(nbytes)
                if nbytes > CHUNK_WARN_BYTES:
                    log.debug("Large chunk: %d bytes", nbytes)

                # (tùy chọn) kiểm thứ tự từ meta
                if sess.pending_meta is not None:
                    meta = sess.pending_meta
                    sess.pending_meta = None  # consume
                    seq = meta.get("seq", 0)
                    if seq != sess.seq_expected:
                        log.debug("Out-of-order chunk: got %d expect %d", seq, sess.seq_expected)
                        sess.seq_expected = seq + 1
                    else:
                        sess.seq_expected += 1

                # ---- DE-OVERLAP ----
                drop = 0 if not sess.seen_first_chunk else min(sess.next_overlap_bytes, nbytes)
                core = raw[drop:]
                if sess.next_overlap_bytes or not sess.seen_first_chunk:
                    print(f"DEBUG deoverlap: first={not sess.seen_first_chunk} drop={drop} keep={len(core)} bytes")

                # tiêu thụ meta cho chunk này
                sess.seen_first_chunk = True
                sess.next_overlap_bytes = 0  # đã dùng

                # ---- nạp core vào buffers ----
                sess.seg_buf.extend(core)
                sess.append_to_rolling(core)

                # ---- rolling decode ----
                try:
                    hyp = await asr_partial_decode_rolling(bytes(sess.roll_buf), sess.sr)
                except Exception as e:
                    log.exception("rolling decode error: %s", e)
                    hyp = ""

                # ---- gating & merge ----
                if hyp and len(_norm(hyp)) >= ROLLING_MIN_HYP_LEN:
                    merged = merge_with_overlap(sess.partial_text, hyp)
                    gain = len(_norm(merged)) - len(_norm(sess.partial_text))
                    stale = (now_ms() - sess.last_emit_ms) >= ROLLING_EMIT_MAX_LAG_MS
                    if gain >= ROLLING_EMIT_MIN_GAIN or stale:
                        sess.partial_text = merged
                        sess.last_emit_ms = now_ms()
                        print(f"DEBUG rolling hyp: {repr(hyp)}")
                        print(f"DEBUG merged text: {repr(sess.partial_text)}")
                        await sess.sendj({"event": "partial", "text": sess.partial_text})

                # ---- VAD finalize ----
                if vad_is_speech(core, sess.sr):
                    sess.last_voice_ms = now_ms()
                else:
                    silence_ms = now_ms() - sess.last_voice_ms
                    too_long = (len(sess.seg_buf) / 2 / sess.sr) >= MAX_SEGMENT_SEC
                    if silence_ms >= FINAL_SILENCE_MS or too_long:
                        try:
                            res = await asr_final_decode(bytes(sess.seg_buf), sess.sr)
                            await sess.sendj({"event": "final", **res})
                            metrics = compute_metrics(sess.seg_buf, sess.sr)
                            await sess.sendj({"event": "metrics", **metrics})
                        except Exception as e:
                            await sess.sendj({"event": "error", "detail": f"decode error: {e}"})
                        finally:
                            # reset utterance state
                            sess.seg_buf.clear()
                            sess.roll_buf.clear()
                            sess.partial_text = ""
                            sess.last_voice_ms = now_ms()
                            sess.last_emit_ms = 0
                            # reset de-overlap cho utterance mới
                            sess.seen_first_chunk = False
                            sess.next_overlap_bytes = 0

            else:
                pass

    except WebSocketDisconnect:
        log.info("Session %s disconnected.", sess.id)
    except Exception as e:
        log.exception("WS error: %s", e)
        try:
            await sess.sendj({"event": "error", "detail": str(e)})
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass
