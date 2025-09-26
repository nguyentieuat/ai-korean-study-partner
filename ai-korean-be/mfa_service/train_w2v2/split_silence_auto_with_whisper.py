# split_silence_auto_with_whisper.py
import os, csv, json, argparse, re, unicodedata as ud
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm
from pydub import AudioSegment
from pydub import silence as pd_silence
import whisper
import torch

# ====== Optional: your phonology rules ======
# Đặt ko_phonology.py (có hàm explain_clean_for_user) cạnh file này nếu muốn sinh text_ref chuẩn surface.
try:
    from ko_phonology import explain_clean_for_user  # (surface_text, rule_tags) = explain_clean_for_user(text)
except Exception:
    explain_clean_for_user = None

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".wma", ".aac"}
JAMO_BLOCK = re.compile(r'[\u1100-\u11FF\u3130-\u318F]')
HANGUL_ONLY = re.compile(r'[\uAC00-\uD7A3]+')

_NUM_RE = re.compile(r"^[0-9]+(\.[0-9]+)?$")
_BAD_TOKENS = {"SIL", "SP", "SILENCE", "NSN", "SPN"}

# ----------------- Utils -----------------
def find_audio_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS]

def ensure_mono16k(seg: AudioSegment) -> AudioSegment:
    """Ép PCM mono 16kHz 16-bit để pydub/FFmpeg ổn định và thống nhất cỡ mẫu."""
    if seg.channels != 1: seg = seg.set_channels(1)
    if seg.frame_rate != 16000: seg = seg.set_frame_rate(16000)
    if seg.sample_width != 2: seg = seg.set_sample_width(2)  # PCM16
    return seg

def norm_nfc(s: str) -> str:
    return ud.normalize("NFC", (s or "").strip())

def make_text_ref(text_orth: str) -> str:
    """Surface text via phonology; fall back sang chính tả nếu không có module."""
    if explain_clean_for_user is None:
        return text_orth
    try:
        surface, _tags = explain_clean_for_user(text_orth)
        return norm_nfc(surface)
    except Exception:
        return text_orth

def jamos_from_surface(text_ref: str) -> List[str]:
    nfd = ud.normalize("NFD", text_ref)
    return [ch for ch in nfd if JAMO_BLOCK.match(ch)]

# ----- Optional G2P (fallback) -----
_G2P = None
def make_phonemes_g2pk2(text_ref: str) -> List[str]:
    """Trả danh sách token phoneme-like từ g2pk2 (cần pip install g2pk2)."""
    global _G2P
    try:
        if _G2P is None:
            from g2pk2 import G2p
            _G2P = G2p()
        out = _G2P(text_ref)
        return list(out) if isinstance(out, str) else out
    except Exception:
        return jamos_from_surface(text_ref)

# ----- MFA dict helpers -----
def sanitize_pron_tokens(tokens: list) -> list:
    """Lọc token không phải âm vị: số (0.99...), rỗng, nhãn im lặng."""
    clean = []
    for t in tokens:
        if not t:
            continue
        if t in _BAD_TOKENS:
            continue
        if _NUM_RE.match(t):   # loại 0.99, 1.2, 2.86...
            continue
        clean.append(t)
    return clean

def load_mfa_dict(path: str):
    """Đọc MFA lexicon: dòng 'WORD PH1 PH2 ...'."""
    lex = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith(";"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            word = parts[0]
            pron = parts[1:]
            pron = sanitize_pron_tokens(pron)
            if not pron:
                continue
            lex.setdefault(word, []).append(pron)
    return lex

def tokenize_hangul(text: str) -> List[str]:
    toks = []
    for tok in (text or "").split():
        toks.extend(HANGUL_ONLY.findall(tok))
    return toks

def phonemes_from_dict(text_orth: str, text_ref: str, lex: dict,
                       dict_on="text", pick="first", fallback="jamo") -> Tuple[List[str], List[str]]:
    """Trả (phoneme list, danh sách OOV)."""
    src = text_ref if dict_on == "text_ref" else text_orth
    tokens = tokenize_hangul(src)
    out, oov = [], []
    for t in tokens:
        vars = lex.get(t)
        if vars:
            chosen = max(vars, key=len) if pick == "longest" else vars[0]
            chosen = sanitize_pron_tokens(chosen)
            out.extend(chosen)
        else:
            oov.append(t)
            if fallback == "g2pk2":
                out.extend(make_phonemes_g2pk2(t))
            elif fallback == "jamo":
                out.extend(jamos_from_surface(t))
            else:
                pass
    return out, oov

# ----------------- Silence-based segmentation -----------------
def detect_by_silence(
    seg: AudioSegment,
    min_silence_len: int = 150,
    silence_offset_db: float = 16.0,
    keep_ms: int = 120,
    join_ms: int = 120,
    min_clip_ms: int = 300,
    max_ms: int = 15000,
) -> List[Tuple[int,int]]:
    """
    Trả về list (start_ms, end_ms) dựa trên rãnh im lặng.
    - min_silence_len: độ dài im lặng tối thiểu để cắt
    - silence_offset_db: ngưỡng = seg.dBFS - offset (càng lớn càng khó coi là im lặng)
    - keep_ms: giữ lại ít im lặng 2 đầu mỗi vùng (đỡ cắt cụt phụ âm cuối)
    - join_ms: nối 2 vùng nói gần nhau
    """
    dbfs = seg.dBFS if seg.dBFS != float("-inf") else -50.0
    silence_thresh = dbfs - silence_offset_db
    regions = pd_silence.detect_nonsilent(
        seg,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh
    )

    # Gộp vùng gần nhau
    merged = []
    for s, e in regions:
        if not merged:
            merged.append([s, e]); continue
        if s - merged[-1][1] <= join_ms:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    # Nới biên & lọc độ dài
    L = len(seg)
    out = []
    for s, e in merged:
        s2 = max(0, s - keep_ms)
        e2 = min(L, e + keep_ms)
        if e2 - s2 >= min_clip_ms:
            cur = s2
            while e2 - cur > max_ms:
                out.append((cur, cur + max_ms))
                cur += max_ms
            out.append((cur, e2))
    return out

# ---- padding (tuỳ chọn) để đảm bảo tối thiểu khi xuất ----
def pad_to_min_bounds(L: int, s: int, e: int, pad_ms: int, min_len_ms: int) -> Tuple[int, int]:
    s2 = max(0, s - pad_ms)
    e2 = min(L, e + pad_ms)
    need = min_len_ms - (e2 - s2)
    if need > 0:
        add_left = min(need // 2, s2)
        add_right = min(need - add_left, L - e2)
        s2 -= add_left
        e2 += add_right
    return s2, e2

def export_clip(src_path: Path, seg: AudioSegment, s: int, e: int, out_root: Path,
                pad_ms: int = 0, ensure_min_ms: int = 0) -> Tuple[Path, int, int]:
    L = len(seg)
    if ensure_min_ms or pad_ms:
        s, e = pad_to_min_bounds(L, s, e, pad_ms, ensure_min_ms or 0)
    rel = src_path.relative_to(ROOT_DIR).with_suffix("")  # mirror tree
    out_dir = out_root / rel
    out_dir.mkdir(parents=True, exist_ok=True)
    clip = seg[s:e]
    out_path = out_dir / f"{src_path.stem}_{s:06d}-{e:06d}.wav"
    clip.export(out_path, format="wav", parameters=["-acodec", "pcm_s16le"])
    return out_path, s, e

# ----------------- IO -----------------
def write_csv_rows(path: Path, rows: List[List]):
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow([
                "source_path","start_ms","end_ms","slice_path","duration_ms",
                "sil_min_len","sil_offset_db","sil_keep_ms","sil_join_ms","max_ms",
                "asr_text","text_ref","jamos_ref","phonemes_ref"
            ])
        w.writerows(rows)

def write_jsonl(path: Path, objs: List[dict]):
    with path.open("a", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

# ----------------- Whisper -----------------
def load_whisper_model(model_name: str, device_pref: Optional[str] = None):
    """device_pref: 'cuda' | 'cpu' | 'auto'."""
    dev = (device_pref or "auto").lower()
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    elif dev == "cuda" and not torch.cuda.is_available():
        print("[INFO] CUDA không khả dụng → fallback CPU.")
        dev = "cpu"
    try:
        return whisper.load_model(model_name, device=dev)
    except RuntimeError as e:
        if "CUDA" in str(e) or "deserialize object on a CUDA device" in str(e):
            print("[INFO] Lỗi khi load CUDA → retry CPU.")
            return whisper.load_model(model_name, device="cpu")
        raise

def transcribe_whisper(model, audio_path: Path, language: Optional[str] = "ko",
                       beam_size: int = 5, temperature: float = 0.0) -> str:
    dev_type = getattr(getattr(model, "device", None), "type", "cpu")
    use_fp16 = True if dev_type == "cuda" else False
    result = model.transcribe(
        str(audio_path),
        language=language,
        task="transcribe",
        verbose=False,
        fp16=use_fp16,
        word_timestamps=False,
        condition_on_previous_text=False,
        beam_size=beam_size,
        best_of=None,
        temperature=temperature
    )
    return (result.get("text") or "").strip()

# ----------------- Main -----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Silence-based segmentation + openai-whisper ASR (+ text_ref/jamo/phoneme)")
    ap.add_argument("--root", type=str, default=r"E:\USE\My_project\source audio")
    ap.add_argument("--out", type=str, default="output_silence")

    # Silence params
    ap.add_argument("--silence_min_len", type=int, default=140, help="ms im lặng tối thiểu để tách")
    ap.add_argument("--silence_offset_db", type=float, default=16.0, help="ngưỡng = dBFS - offset")
    ap.add_argument("--silence_keep_ms", type=int, default=80, help="đệm im lặng giữ lại 2 đầu mỗi vùng")
    ap.add_argument("--silence_join_ms", type=int, default=120, help="nối 2 vùng nói gần nhau")
    ap.add_argument("--min_clip_ms", type=int, default=400, help="bỏ clip quá ngắn sau tách")
    ap.add_argument("--max_ms", type=int, default=15000, help="chẻ nếu clip quá dài")

    # Export padding (mặc định 0 để vết cắt “sạch” theo silence)
    ap.add_argument("--pad_ms", type=int, default=0, help="đệm 2 đầu khi xuất (ms)")
    ap.add_argument("--ensure_min_ms", type=int, default=0, help="đảm bảo clip >= giá trị này (ms)")

    # ASR (openai-whisper)
    ap.add_argument("--do_asr", action="store_true")
    ap.add_argument("--whisper_model", type=str, default="small", help="tiny|base|small|medium|large|large-v3")
    ap.add_argument("--whisper_device", type=str, default="auto", help="auto|cuda|cpu")
    ap.add_argument("--whisper_lang", type=str, default="ko")
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.0)

    # Surface/phoneme flags
    ap.add_argument("--emit_surface", action="store_true", help="Sinh text_ref, jamos_ref, phonemes_ref")

    ap.add_argument("--phonemes_mode", type=str, default="none",
                    choices=["none","g2pk2","dict"],
                    help="none=jamo fallback; g2pk2=G2P; dict=MFA lexicon")

    # Dict options (for phonemes_mode=dict)
    ap.add_argument("--dict_path", type=str, default=None, help="Đường dẫn MFA dict (vd: korean_mfa.dict)")
    ap.add_argument("--dict_on", type=str, default="text", choices=["text","text_ref"],
                    help="Áp dict lên text (chính tả) hay text_ref (bề mặt)")
    ap.add_argument("--dict_pick", type=str, default="first", choices=["first","longest"],
                    help="Chọn biến thể phát âm: first|longest")
    ap.add_argument("--dict_fallback", type=str, default="jamo", choices=["jamo","g2pk2","none"],
                    help="Khi OOV: jamo|g2pk2|none")
    ap.add_argument("--oov_log", type=str, default="oov_words.txt", help="log từ OOV")

    args = ap.parse_args()

    global ROOT_DIR
    ROOT_DIR = Path(args.root)
    OUT_ROOT = Path(args.out)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_ROOT / "manifest.csv"
    jsonl_path = OUT_ROOT / "slices_asr.jsonl"

    files = find_audio_files(ROOT_DIR)
    if not files:
        print("Không thấy file audio.")
        raise SystemExit

    # Load Whisper nếu bật do_asr
    asr_model = None
    if args.do_asr:
        print(f"[ASR] openai-whisper model={args.whisper_model}, device={args.whisper_device}")
        asr_model = load_whisper_model(args.whisper_model, device_pref=args.whisper_device)

    # Load lexicon nếu cần
    lex = None
    oov_set = set()
    if args.emit_surface and args.phonemes_mode == "dict":
        if not args.dict_path:
            raise SystemExit("[ERROR] --phonemes_mode dict cần --dict_path")
        print(f"[DICT] Loading lexicon from: {args.dict_path}")
        lex = load_mfa_dict(args.dict_path)
        print(f"[DICT] Entries: {len(lex)}")

    total = 0
    for p in tqdm(files, desc="Processing", unit="file"):
        try:
            seg0 = AudioSegment.from_file(p)
            seg = ensure_mono16k(seg0)

            ranges = detect_by_silence(
                seg,
                min_silence_len=args.silence_min_len,
                silence_offset_db=args.silence_offset_db,
                keep_ms=args.silence_keep_ms,
                join_ms=args.silence_join_ms,
                min_clip_ms=args.min_clip_ms,
                max_ms=args.max_ms
            )
            if not ranges:
                continue

            rows, objs = [], []
            for s, e in ranges:
                out_wav, s2, e2 = export_clip(
                    p, seg, s, e, OUT_ROOT,
                    pad_ms=args.pad_ms, ensure_min_ms=args.ensure_min_ms
                )

                # ASR
                text = ""
                if asr_model is not None:
                    try:
                        text = transcribe_whisper(
                            asr_model, out_wav,
                            language=args.whisper_lang if args.whisper_lang else None,
                            beam_size=args.beam_size,
                            temperature=args.temperature
                        )
                    except Exception as ex:
                        print(f"[ASR WARN] {out_wav}: {ex}")
                        text = ""

                text_orth = norm_nfc(text)
                text_ref, jamo_ref, phon_ref = "", [], []

                if args.emit_surface:
                    text_ref = make_text_ref(text_orth)
                    jamo_ref = jamos_from_surface(text_ref)
                    if args.phonemes_mode == "dict" and lex is not None:
                        phon_ref, oov_words = phonemes_from_dict(
                            text_orth, text_ref, lex,
                            dict_on=args.dict_on,
                            pick=args.dict_pick,
                            fallback=args.dict_fallback
                        )
                        for w in oov_words: oov_set.add(w)
                    elif args.phonemes_mode == "g2pk2":
                        phon_ref = make_phonemes_g2pk2(text_ref)
                    else:
                        phon_ref = jamo_ref[:]  # safe default

                dur = e2 - s2
                rows.append([
                    str(p), s2, e2, str(out_wav), dur,
                    args.silence_min_len, args.silence_offset_db, args.silence_keep_ms, args.silence_join_ms, args.max_ms,
                    text_orth,
                    text_ref, " ".join(jamo_ref), " ".join(map(str, phon_ref))
                ])
                objs.append({
                    "audio": str(out_wav),
                    "duration_ms": dur,
                    "start_ms": s2, "end_ms": e2,
                    "text": text_orth,
                    "text_ref": text_ref,
                    "jamos_ref": jamo_ref,
                    "phonemes_ref": phon_ref,
                    "source": str(p),
                    "kind": "slice",
                    "params": {
                        "silence_min_len": args.silence_min_len,
                        "silence_offset_db": args.silence_offset_db,
                        "silence_keep_ms": args.silence_keep_ms,
                        "silence_join_ms": args.silence_join_ms,
                        "max_ms": args.max_ms,
                        "pad_ms": args.pad_ms,
                        "ensure_min_ms": args.ensure_min_ms
                    }
                })
                total += 1

            write_csv_rows(csv_path, rows)
            write_jsonl(jsonl_path, objs)

        except Exception as ex:
            print(f"[WARN] {p}: {ex}")

    # Write OOV log
    if lex is not None and oov_set and args.oov_log:
        oov_path = OUT_ROOT / args.oov_log
        with open(oov_path, "w", encoding="utf-8") as f:
            for w in sorted(oov_set):
                f.write(w + "\n")
        print(f"[DICT] OOV words: {len(oov_set)} → {oov_path.resolve()}")

    print(f"XONG. Tổng đoạn: {total}")
    print(f"- CSV:    {csv_path.resolve()}")
    print(f"- JSONL:  {jsonl_path.resolve()}")
    print(f"- Output: {OUT_ROOT.resolve()}")
