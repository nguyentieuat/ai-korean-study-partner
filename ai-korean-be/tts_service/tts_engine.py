import os, io, re, hashlib
from threading import Lock
from typing import Tuple, List, Optional
import numpy as np, soundfile as sf, librosa, torch
from transformers import pipeline
from uroman import Uroman

# Model & runtime
MODEL_KOR = os.getenv("MODEL_KOR", "facebook/mms-tts-kor")
DEVICE_ID = 0 if torch.cuda.is_available() else -1
DTYPE = torch.float16 if (torch.cuda.is_available() and os.getenv("TTS_FP16","1")=="1") else torch.float32
CACHE_DIR = os.getenv("CACHE_DIR", "/data/tts_cache"); os.makedirs(CACHE_DIR, exist_ok=True)
MAX_CHARS_PER_SENT = int(os.getenv("TTS_MAX_CHARS_PER_SENT","300"))

# Romanizer — ép dùng luôn
_URO = Uroman()
def _romanize_ko(t: str) -> str:
    r = _URO.romanize_string(t)
    return r

# Lazy pipeline
_PIPE = None; _LOCK = Lock()
def get_pipe():
    global _PIPE
    with _LOCK:
        if _PIPE is None:
            _PIPE = pipeline("text-to-speech", model=MODEL_KOR, torch_dtype=DTYPE, device=DEVICE_ID)
    return _PIPE

# Split câu (giữ dấu) + cắt mềm
_SPLIT = re.compile(r"([.!?…]|[。？！]|[.?!])")
def split_sentences(text: str) -> List[str]:
    parts = [p.strip() for p in _SPLIT.split(text)]
    sents = []
    for i in range(0, len(parts), 2):
        s = parts[i]
        if i+1 < len(parts) and parts[i+1]: s += parts[i+1]
        if s: sents.append(s)
    if not sents: sents = [text.strip()]
    out = []
    for s in sents:
        if len(s) <= MAX_CHARS_PER_SENT:
            out.append(s); continue
        cur, buf = 0, []
        for tok in s.split(" "):
            if cur + len(tok) + 1 > MAX_CHARS_PER_SENT and buf:
                out.append(" ".join(buf)); buf=[tok]; cur=len(tok)
            else:
                buf.append(tok); cur += len(tok) + 1
        if buf: out.append(" ".join(buf))
    return out

def _hash(text:str, slow:float, pause_ms:int)->str:
    h=hashlib.sha256(); h.update(f"{slow}|||{pause_ms}|||{text}".encode("utf-8")); return h.hexdigest()

def _write_wav_bytes(audio: np.ndarray, sr:int)->bytes:
    if audio.size and np.max(np.abs(audio))>1.0: audio = audio/(np.max(np.abs(audio))+1e-8)
    buf=io.BytesIO(); sf.write(buf, audio.astype(np.float32, copy=False), sr, format="WAV", subtype="PCM_16"); buf.seek(0); return buf.read()

def synthesize(text:str, slow_factor:float=0.9, pause_ms:int=400, use_cache:bool=True) -> Tuple[bytes,int,str]:
    if not isinstance(text,str) or not text.strip(): raise ValueError("text rỗng.")
    slow=float(slow_factor); 
    if not (0.5<=slow<=1.5): raise ValueError("slow_factor phải trong [0.5, 1.5].")
    pause=max(0,min(int(pause_ms),5000))

    # ÉP ROMANIZE TOÀN BỘ trước khi split
    text_ro = _romanize_ko(text)
    key=_hash(text_ro, slow, pause); cpath=os.path.join(CACHE_DIR, f"{key}.wav")
    if use_cache and os.path.exists(cpath):
        with open(cpath,"rb") as f: return f.read(),16000,key

    pipe=get_pipe(); sents=split_sentences(text_ro)
    wavs=[]; sr_out=None
    for s in sents:
        out=pipe(s)                       # MMS-KOR single-speaker
        audio=np.asarray(out["audio"], dtype=np.float32)
        sr=int(out["sampling_rate"]); 
        if sr_out is None: sr_out=sr
        if abs(slow-1.0)>1e-3: audio=librosa.effects.time_stretch(audio, rate=slow)
        wavs.append(audio)
        if pause>0: wavs.append(np.zeros(int(sr*(pause/1000.0)), dtype=np.float32))

    audio_all=np.concatenate(wavs) if wavs else np.zeros(1, dtype=np.float32)
    sr_out=sr_out or 16000; wav_bytes=_write_wav_bytes(audio_all, sr_out)
    if use_cache:
        try: 
            with open(cpath,"wb") as f: f.write(wav_bytes)
        except Exception: pass
    return wav_bytes, sr_out, key
