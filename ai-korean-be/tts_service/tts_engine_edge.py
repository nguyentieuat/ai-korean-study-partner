# tts_engine.py — Edge-TTS (Korean), m/f, cache, slow_factor & pause
# Có fallback: nếu API Python không trả audio, sẽ gọi CLI: python -m edge_tts
# Yêu cầu:
#   pip install edge-tts pydub soundfile
#   ffmpeg trong PATH để xuất WAV & chèn pause
# Trả về: (audio_bytes, SAMPLE_RATE, cache_key)
#  - Có pydub+ffmpeg: WAV 24k, có chèn pause giữa câu
#  - Không có pydub/ffmpeg: MP3 (không chèn pause), /api/tts/file sẽ trả mp3

import os, io, re, hashlib, asyncio, subprocess, sys, tempfile
from typing import Tuple, List, Optional
from pathlib import Path

MODEL_KOR = "edge-tts-ko"
DEVICE_ID = -1
CACHE_DIR = os.getenv("CACHE_DIR", os.path.abspath("./tts_cache"))
os.makedirs(CACHE_DIR, exist_ok=True)

VOICE_MAP = {
    "m": os.getenv("EDGE_VOICE_M", "ko-KR-InJoonNeural"),   # Nam
    "f": os.getenv("EDGE_VOICE_F", "ko-KR-SunHiNeural"),    # Nữ
}
DEFAULT_SPEAKER = os.getenv("EDGE_DEFAULT_SPEAKER", "f")
SAMPLE_RATE = 24000

FALLBACK = {
    "f": [
        VOICE_MAP["f"],
        "ko-KR-JiMinNeural",
        "ko-KR-SeoHyeonNeural",
        "ko-KR-YuJinNeural",
        "ko-KR-SoonBokNeural",
    ],
    "m": [
        VOICE_MAP["m"],
        "ko-KR-BongJinNeural",
        "ko-KR-GookMinNeural",
        "ko-KR-HyunsuNeural",
        "ko-KR-HyunsuMultilingualNeural",
    ],
}

# pydub để xuất WAV & ghép im lặng
try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except Exception:
    HAS_PYDUB = False

def _rate_from_slow(slow_factor: float) -> str:
    sf = float(slow_factor)
    sf = 0.5 if sf < 0.5 else (1.5 if sf > 1.5 else sf)
    pct = int(round((sf - 1.0) * 100.0))
    return f"{pct:+d}%"

_SPLIT = re.compile(r"([.!?…]|[。？！])")
def _split_sentences(text: str) -> List[str]:
    parts = [p.strip() for p in _SPLIT.split(text)]
    sents = []
    for i in range(0, len(parts), 2):
        s = parts[i]
        if i + 1 < len(parts) and parts[i+1]:
            s += parts[i+1]
        if s:
            sents.append(s)
    return sents or [text.strip()]

def _hash_key(text: str, slow: float, pause_ms: int, speaker: str) -> str:
    h = hashlib.sha256()
    h.update(f"edge|{speaker}|{slow}|{pause_ms}|{text}".encode("utf-8"))
    return h.hexdigest()

# ---------- Edge-TTS qua API Python (stream MP3) ----------
async def _edge_mp3_async(text: str, voice: str, rate: str) -> bytes:
    import edge_tts
    com = edge_tts.Communicate(text, voice=voice, rate=rate, pitch="+0Hz")
    buf = io.BytesIO(); got = False
    async for chunk in com.stream():  # không truyền 'format'
        if chunk["type"] == "audio":
            buf.write(chunk["data"]); got = True
    if got and buf.tell() > 0:
        return buf.getvalue()

    # fallback: gọi save() ra file tạm
    com2 = edge_tts.Communicate(text, voice=voice, rate=rate, pitch="+0Hz")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp_path = tmp.name
    try:
        await com2.save(tmp_path)
        data = Path(tmp_path).read_bytes()
        if data:
            return data
        raise RuntimeError("No audio was received (save).")
    finally:
        try: os.remove(tmp_path)
        except: pass

# ---------- Edge-TTS qua CLI (ultimate fallback) ----------
def _edge_cli_mp3(text: str, voice: str, rate: str) -> bytes:
    """
    Fallback dùng CLI: ưu tiên interpreter do người dùng chỉ định (EDGE_TTS_PY),
    truyền proxy nếu có, ghi text ra file tạm (--file=...), thử với --rate rồi
    thử lại không --rate nếu lần 1 không có audio.
    """
    import tempfile, subprocess, sys, os
    from pathlib import Path

    pyexe = os.getenv("EDGE_TTS_PY") or sys.executable
    proxy = os.getenv("EDGE_TTS_PROXY") or os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")

    # ghi text ra file để tránh lỗi quote
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tf:
        tf.write(text)
        txt_path = tf.name

    def run_one(use_rate: bool) -> bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as mf:
            mp3_path = mf.name
        try:
            cmd = [
                pyexe, "-m", "edge_tts",
                f"--voice={voice}",
                f"--file={txt_path}",
                f"--write-media={mp3_path}",
            ]
            if proxy:
                cmd.append(f"--proxy={proxy}")
            if use_rate:
                cmd.append(f"--rate={rate}")  # dạng --rate=-10%
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, timeout=180)
            if r.returncode != 0:
                raise RuntimeError(f"edge-tts CLI failed (rc={r.returncode}): {r.stderr or r.stdout}")
            data = Path(mp3_path).read_bytes()
            if not data:
                raise RuntimeError("edge-tts CLI wrote empty file.")
            return data
        finally:
            try: os.remove(mp3_path)
            except: pass

    try:
        try:
            return run_one(True)   # thử có --rate
        except Exception:
            return run_one(False)  # fallback: bỏ --rate
    finally:
        try: os.remove(txt_path)
        except: pass

# ---------- Xử lý WAV (pydub) ----------
def _mp3_to_wav_bytes(mp3_bytes: bytes, sr: int) -> bytes:
    seg = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
    seg = seg.set_frame_rate(sr).set_channels(1).set_sample_width(2)
    out = io.BytesIO()
    seg.export(out, format="wav")
    return out.getvalue()

def _concat_with_silence_wav(sent_wavs: List[bytes], pause_ms: int, sr: int) -> bytes:
    if not sent_wavs:
        seg = AudioSegment.silent(duration=50, frame_rate=sr).set_channels(1).set_sample_width(2)
        out = io.BytesIO(); seg.export(out, format="wav"); return out.getvalue()
    silence = AudioSegment.silent(duration=max(pause_ms, 0), frame_rate=sr).set_channels(1).set_sample_width(2)
    full = None
    for i, wb in enumerate(sent_wavs):
        seg = AudioSegment.from_file(io.BytesIO(wb), format="wav").set_frame_rate(sr).set_channels(1).set_sample_width(2)
        full = seg if full is None else (full + silence + seg)
    out = io.BytesIO(); full.export(out, format="wav"); return out.getvalue()

# ---------- Public API ----------
def synthesize(
    text: str,
    slow_factor: float = 0.9,
    pause_ms: int = 400,
    use_cache: bool = True,
    speaker: Optional[str] = None,
) -> Tuple[bytes, int, str]:
    # sanitize
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text rỗng.")
    clean_text = re.sub(r"\s+", " ", text).strip()

    spk = (speaker or DEFAULT_SPEAKER).lower()
    cand_voices = FALLBACK.get(spk, [VOICE_MAP.get(spk, VOICE_MAP["f"])])
    rate = _rate_from_slow(slow_factor)

    key = _hash_key(clean_text, slow_factor, pause_ms, spk)
    cpath_wav = os.path.join(CACHE_DIR, f"{key}.wav")
    cpath_mp3 = os.path.join(CACHE_DIR, f"{key}.mp3")

    # cache hit
    if use_cache and os.path.exists(cpath_wav):
        return Path(cpath_wav).read_bytes(), SAMPLE_RATE, key
    if use_cache and os.path.exists(cpath_mp3):
        return Path(cpath_mp3).read_bytes(), SAMPLE_RATE, key

    # Có pydub/ffmpeg → synth theo câu + chèn pause → WAV
    if HAS_PYDUB:
        sents = _split_sentences(clean_text)
        last_err: Optional[Exception] = None
        for v in cand_voices:
            try:
                sent_wavs: List[bytes] = []
                # 1) thử API Python
                ok = True
                for s in sents:
                    try:
                        mp3 = asyncio.run(_edge_mp3_async(s, v, rate))
                    except Exception:
                        ok = False; break
                    wav = _mp3_to_wav_bytes(mp3, SAMPLE_RATE)
                    sent_wavs.append(wav)
                if not ok or not sent_wavs:
                    # 2) fallback từng câu bằng CLI
                    sent_wavs = []
                    for s in sents:
                        mp3 = _edge_cli_mp3(s, v, rate)
                        wav = _mp3_to_wav_bytes(mp3, SAMPLE_RATE)
                        sent_wavs.append(wav)

                wav_bytes = _concat_with_silence_wav(sent_wavs, pause_ms, SAMPLE_RATE)
                if use_cache:
                    Path(cpath_wav).write_bytes(wav_bytes)
                return wav_bytes, SAMPLE_RATE, key
            except Exception as e:
                last_err = e; continue
        raise RuntimeError(f"No audio was received from voices {cand_voices}: {last_err}")

    # Không có pydub/ffmpeg → synth một lần (MP3), không chèn pause
    last_err: Optional[Exception] = None
    for v in cand_voices:
        try:
            # 1) API Python
            try:
                mp3 = asyncio.run(_edge_mp3_async(clean_text, v, rate))
            except Exception:
                # 2) CLI fallback
                mp3 = _edge_cli_mp3(clean_text, v, rate)

            if use_cache:
                Path(cpath_mp3).write_bytes(mp3)
            return mp3, SAMPLE_RATE, key
        except Exception as e:
            last_err = e; continue
    raise RuntimeError(f"No audio was received from voices {cand_voices}: {last_err}")
