# %%writefile run_asr_phonology.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_asr_phonology.py (RESUMABLE + FASTER + MULTIPROCESS + MULTI-GPU + JAMO-SPACE + NUMBER/UNIT VERBALIZE)

- Engine: openai-whisper hoặc faster-whisper (CTranslate2)
- Resume an toàn qua CSV/JSONL
- Shard theo chỉ số/tổng số shard
- Multiprocessing CPU & Multi-GPU (mỗi GPU một tiến trình)
- Thay prefix đường dẫn manifest: --path_replace_old/--path_replace_new
- Jamo: tuỳ chọn giữ ranh giới từ bằng token (▁), giữ ASCII/punct nếu muốn
- Số/Đơn vị: **ko_phonology trước → verbalize số/đơn vị sau** (auto/sino/native) vào text_ref/orth theo lựa chọn
"""

import os, csv, json, argparse, math, tempfile, unicodedata as ud, re, string
from pathlib import Path
from typing import List, Optional, Set, Dict, Any, Tuple
from tqdm import tqdm
import multiprocessing as mp

# Audio I/O
from pydub import AudioSegment
from pydub import effects as pd_effects

# Optional ko_phonology
try:
    from ko_phonology import explain_clean_for_user, normalize_clean  # type: ignore
except Exception:
    explain_clean_for_user = None
    normalize_clean = None

# Core ASR libs
import torch
import whisper  # openai-whisper

# Optional fast duration
try:
    import soundfile as sf
    _HAS_SF = True
except Exception:
    _HAS_SF = False

from concurrent.futures import ProcessPoolExecutor, as_completed

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".wma", ".aac"}

# ====================== Number reading helpers ======================

_SINO_DIG = ["영","일","이","삼","사","오","육","칠","팔","구"]
_SINO_SMALL = ["","십","백","천"]
_SINO_BIG = [("조",10**12),("억",10**8),("만",10**4)]
_NATIVE_ONES = {
    1:"한", 2:"두", 3:"세", 4:"네", 5:"다섯", 6:"여섯", 7:"일곱", 8:"여덟", 9:"아홉",
    10:"열", 20:"스물", 30:"서른", 40:"마흔", 50:"쉰", 60:"예순", 70:"일흔", 80:"여든", 90:"아흔"
}
def _native_number(n: int) -> str:
    if n <= 0: return "영"
    if n in _NATIVE_ONES: return _NATIVE_ONES[n]
    tens = (n//10)*10
    ones = n%10
    return f"{_NATIVE_ONES.get(tens, '')}{_NATIVE_ONES.get(ones, '')}"

def _read_chunk_4(n: int) -> str:
    s = []
    for i in range(3,-1,-1):
        unit = 10**i
        d = (n // unit) % 10
        if d == 0: continue
        if i >= 1:
            if d == 1: s.append(_SINO_SMALL[i])
            else: s.append(_SINO_DIG[d] + _SINO_SMALL[i])
        else:
            s.append(_SINO_DIG[d])
    return "".join(s) if s else ""

def read_number_sino(num_str: str) -> str:
    s = num_str.replace(",", "")
    if s.count(".") > 1: return s
    if "." in s:
        int_part, frac_part = s.split(".", 1)
    else:
        int_part, frac_part = s, ""
    try:
        n = int(int_part) if int_part else 0
    except Exception:
        return num_str
    if n == 0:
        int_read = "영"
    else:
        parts = []; rest = n
        for big_name, big_val in _SINO_BIG:
            q = rest // big_val
            if q:
                parts.append(_read_chunk_4(q) + big_name); rest %= big_val
        small = _read_chunk_4(rest)
        if small: parts.append(small)
        int_read = "".join(parts) if parts else "영"
    if frac_part:
        frac_read = "점 " + " ".join(_SINO_DIG[int(d)] if d.isdigit() else d for d in frac_part)
        return f"{int_read} {frac_read}"
    return int_read

def read_number_native_hours(n_str: str) -> str:
    try:
        n = int(n_str.replace(",", ""))
    except Exception:
        return n_str
    if n == 20: head = "스무"
    elif n in (1,2,3,4): head = _native_number(n)
    else: head = _native_number(n)
    return f"{head} 시간"

_UNIT_MAP: Dict[str, Tuple[str,str]] = {
    "%": ("퍼센트", "sino"),
    "mm": ("밀리미터", "sino"),
    "cm": ("센티미터", "sino"),
    "m": ("미터", "sino"),
    "km": ("킬로미터", "sino"),
    "kg": ("킬로그램", "sino"),
    "g": ("그램", "sino"),
    "t": ("톤", "sino"),
    "L": ("리터", "sino"),
    "l": ("리터", "sino"),
    "ml": ("밀리리터", "sino"),
    "μl": ("마이크로리터", "sino"),
    "ul": ("마이크로리터", "sino"),
    "h": ("시간", "native_hours"),
    "s": ("초", "sino"),
    "ms": ("밀리초", "sino"),
    "μs": ("마이크로초", "sino"),
    "us": ("마이크로초", "sino"),
}

_num_pat = re.compile(
    r"""
    (?P<num>
        (?:
            \d{1,3}(?:,\d{3})+
            |
            \d+
        )
        (?:\.\d+)?
    )
    \s*
    (?P<unit>
        %|mm|cm|km|m|kg|g|t|L|l|ml|μl|ul|h|s|ms|μs|us
    )?
    """,
    re.VERBOSE
)

def verbalize_numbers_ko(text: str, mode: str = "auto") -> str:
    def repl(m: re.Match) -> str:
        num = m.group("num")
        unit = m.group("unit")
        read_mode = mode
        unit_word = ""
        if unit:
            unit_word, u_mode = _UNIT_MAP.get(unit, ("", "sino"))
            if mode == "auto":
                read_mode = "native" if u_mode == "native_hours" else "sino"
            elif mode == "native" and u_mode != "native_hours":
                read_mode = "sino"
        else:
            if mode == "auto":
                read_mode = "sino"
        if read_mode == "native":
            spoken = read_number_native_hours(num) if (unit and _UNIT_MAP.get(unit, ("",""))[1]=="native_hours") else read_number_sino(num)
        else:
            spoken = read_number_sino(num)
        if unit_word:
            return f"{spoken} {unit_word}"
        return spoken
    return _num_pat.sub(repl, text)

# ====================== Utils ======================

def find_audio_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS]

def ensure_mono16k(seg: AudioSegment) -> AudioSegment:
    if seg.channels != 1: seg = seg.set_channels(1)
    if seg.frame_rate != 16000: seg = seg.set_frame_rate(16000)
    if seg.sample_width != 2: seg = seg.set_sample_width(2)  # PCM16
    return seg

def nfc(s: str) -> str:
    return ud.normalize("NFC", (s or "").strip())

_PUNCT_SET = set(string.punctuation) | {"…","—","–","·","•","’","‘","“","”","、","，","。","：","；","？","！","（","）","【","】","《","》","〈","〉"}
def jamos_from_text(
    s: str,
    keep_space: bool = False,
    space_token: str = "▁",
    keep_ascii: bool = False,
    keep_punct: bool = False,
    ascii_lower: bool = False
):
    out: List[str] = []
    s = s or ""
    for ch in s:
        if ch.isspace():
            if keep_space and (not out or out[-1] != space_token):
                out.append(space_token)
            continue
        if '\uAC00' <= ch <= '\uD7A3':
            for c in ud.normalize("NFD", ch):
                out.append(c)
            continue
        if '\u1100' <= ch <= '\u11FF' or '\u3130' <= ch <= '\u318F':
            out.append(ch); continue
        if keep_ascii and ch.isascii() and (ch.isalpha() or ch.isdigit()):
            out.append(ch.lower() if ascii_lower else ch); continue
        if keep_punct and (ch in _PUNCT_SET):
            out.append(ch); continue
    return out

def fast_duration_ms(p: Path) -> Optional[int]:
    try:
        import soundfile as sf
        d = sf.info(str(p)).duration
        if d and d > 0: return int(d * 1000)
    except Exception:
        pass
    try:
        return len(AudioSegment.from_file(p))
    except Exception:
        return None

# =================== Model loaders =================

def load_openai_model(model_name: str, device_pref: str = "auto", device_index: int = -1):
    dev = (device_pref or "auto").lower()
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    elif dev == "cuda" and not torch.cuda.is_available():
        print("[INFO] CUDA not available → fallback CPU.")
        dev = "cpu"
    if dev == "cuda":
        try:
            torch.cuda.set_device(0 if device_index is not None and device_index >= 0 else 0)
            print(f"[GPU] openai-whisper using local CUDA device #0 (visible={os.environ.get('CUDA_VISIBLE_DEVICES','all')})")
        except Exception as e:
            print(f"[WARN] torch.cuda.set_device(0) failed: {e}")
    try:
        return whisper.load_model(model_name, device=dev)
    except RuntimeError as e:
        if "CUDA" in str(e) or "deserialize object on a CUDA device" in str(e):
            print("[INFO] Error when loading CUDA model → retry CPU.")
            return whisper.load_model(model_name, device="cpu")
        raise

def load_faster_model(model_name: str, device_pref: str, compute_type: str, device_index: int = -1,
                      num_workers: int = 1, cpu_threads: int = 0):
    from faster_whisper import WhisperModel
    dev = (device_pref or "auto").lower()
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    idx = 0 if dev == "cuda" else -1
    if dev == "cuda":
        print(f"[GPU] faster-whisper on local CUDA device_index={idx} (visible={os.environ.get('CUDA_VISIBLE_DEVICES','all')})")
    return WhisperModel(
        model_name,
        device=dev,
        device_index=(idx if idx >= 0 else None),
        compute_type=compute_type,
        num_workers=max(1, int(num_workers)),
        cpu_threads=max(0, int(cpu_threads)),
    )

# ===================== Transcribers =====================

def transcribe_openai(model, path: Path, language: Optional[str], beam_size: int, temperature: float,
                      nospeech_thresh: float, avglogprob_thresh: float, comp_ratio_max: float) -> str:
    dev_type = getattr(getattr(model, "device", None), "type", "cpu")
    use_fp16 = True if dev_type == "cuda" else False
    result = model.transcribe(
        str(path),
        language=language, task="transcribe", verbose=False, fp16=use_fp16,
        word_timestamps=False, condition_on_previous_text=False,
        beam_size=beam_size, best_of=None, temperature=temperature
    )
    segs = result.get("segments", [])
    pieces = []
    for seg in segs:
        no_speech = float(seg.get("no_speech_prob", 0.0))
        avg_lp    = float(seg.get("avg_logprob", 0.0))
        comp      = float(seg.get("compression_ratio", 0.0))
        if (no_speech > nospeech_thresh and avg_lp < avglogprob_thresh) or (comp > comp_ratio_max):
            continue
        txt = (seg.get("text") or "").strip()
        if txt: pieces.append(txt)
    text = " ".join(pieces).strip()
    return text or (result.get("text") or "").strip()

def transcribe_faster(model, path: Path, language: Optional[str], beam_size: int, temperature: float, vad: bool) -> str:
    segments, _info = model.transcribe(
        str(path),
        language=language,
        vad_filter=vad,
        beam_size=max(1, beam_size),
        temperature=temperature
    )
    return "".join(s.text for s in segments).strip()

# ====================== I/O helpers ======================

def rows_from_manifest(manifest_csv: Path, old=None, new=None) -> List[Path]:
    paths = []
    with manifest_csv.open("r", encoding="utf-8", errors="ignore") as f:
        r = csv.DictReader(f)
        cand_cols = [c for c in ["slice_path","path","audio","wav","file","filepath"] if c in (r.fieldnames or [])]
        if not cand_cols:
            raise SystemExit("Manifest must have one of columns: slice_path/path/audio/wav/file/filepath")
        col = cand_cols[0]
        for row in r:
            raw = (row.get(col) or "").strip().replace("\\", "/")
            if not raw: continue
            if old and new and raw.startswith(old):
                raw = new + raw[len(old):]
            p = Path(raw)
            if p.exists(): paths.append(p)
    return paths

def load_done_set(csv_path: Path, jsonl_path: Path) -> Set[str]:
    done: Set[str] = set()
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8", errors="ignore") as jf:
            for line in jf:
                try:
                    obj = json.loads(line)
                    a = obj.get("audio")
                    if isinstance(a, str) and a:
                        done.add(a)
                except Exception:
                    continue
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", errors="ignore") as cf:
            rd = csv.DictReader(cf)
            if rd.fieldnames and "audio" in rd.fieldnames:
                for row in rd:
                    a = row.get("audio")
                    if isinstance(a, str) and a:
                        done.add(a)
            else:
                cf.seek(0)
                for i, row in enumerate(csv.reader(cf)):
                    if i == 0: continue
                    try:
                        a = row[0]
                        if a: done.add(a)
                    except Exception:
                        pass
    return done

def partition_paths(all_paths: List[Path], shard_idx: int, shard_total: int) -> List[Path]:
    if shard_total <= 1: return all_paths
    if not (0 <= shard_idx < shard_total):
        raise SystemExit(f"Bad shard params: shard_idx={shard_idx}, shard_total={shard_total}")
    per = math.ceil(len(all_paths) / shard_total)
    return all_paths[shard_idx*per : (shard_idx+1)*per]

# ====================== Pipeline helpers ======================

def _surface_pipeline(ref_or_orth_src: str, args_dict: Dict[str, Any]) -> Tuple[str, Optional[List[str]]]:
    text_ref = ref_or_orth_src
    notes: Optional[List[str]] = None
    if explain_clean_for_user is not None:
        try:
            text_ref, events = explain_clean_for_user(ref_or_orth_src)
            text_ref = nfc(text_ref)
            notes = list(dict.fromkeys([e.get("note","") for e in events if isinstance(e.get("note",""), str) and e.get("note","").strip()]))
        except Exception as ex:
            text_ref = ref_or_orth_src; notes = [f"ko_phonology_error: {ex}"]
    elif normalize_clean is not None:
        try:
            text_ref, _tags = normalize_clean(ref_or_orth_src); text_ref = nfc(text_ref)
        except Exception as ex:
            text_ref = ref_or_orth_src; notes = [f"ko_phonology_error: {ex}"]
    return text_ref, notes

def _pipeline_surface_then_verbalize(text_orth: str, args_dict: Dict[str, Any]) -> Tuple[str, str, Optional[List[str]]]:
    """
    Apply ko_phonology first (to produce base text_ref), THEN verbalize numbers/units.
    Returns: (orth_out, text_ref_out, notes)
    """
    orth_out = text_orth
    notes: Optional[List[str]] = None

    # 1) ko_phonology → base_ref
    if args_dict.get("emit_surface", False) and (text_orth.strip()):
        base_ref, notes = _surface_pipeline(text_orth, args_dict)
    else:
        base_ref = text_orth

    text_ref_out = base_ref

    # 2) verbalize numbers/units AFTER phonology
    if args_dict.get("verbalize_numbers", False):
        mode = args_dict.get("num_reading","auto")
        target = args_dict.get("verbalize_target","ref")  # orth|ref|both
        if target in ("orth","both"):
            orth_out = verbalize_numbers_ko(orth_out, mode=mode)
        if target in ("ref","both"):
            text_ref_out = verbalize_numbers_ko(text_ref_out, mode=mode)

    return orth_out, text_ref_out, notes

# ====================== Worker (CPU/GPU) ======================

def _pin_gpu_env(device_index: int):
    if device_index is not None and device_index >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)
        for k in ["CUDA_DEVICE_ORDER"]:
            os.environ.pop(k, None)
        print(f"[GPU/worker] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

def _worker_run(file_list: List[Path], args_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    engine = args_dict["engine"]
    dev_idx = args_dict.get("device_index", -1)

    _pin_gpu_env(dev_idx)

    if engine == "faster":
        model = load_faster_model(
            args_dict["whisper_model"], args_dict["whisper_device"], args_dict["compute_type"], device_index=0,
            num_workers=args_dict.get("fw_num_workers", 1), cpu_threads=args_dict.get("fw_cpu_threads", 0)
        )
    else:
        model = load_openai_model(args_dict["whisper_model"], args_dict["whisper_device"], device_index=0)

    try:
        import torch
        print(f"[worker@GPU{dev_idx}] torch device idx={torch.cuda.current_device()}, "
              f"name={torch.cuda.get_device_name(torch.cuda.current_device())}")
    except Exception:
        pass

    processed = 0
    for p in file_list:
        try:
            tmp_norm_path = None
            audio_for_asr = p
            if args_dict["normalize"]:
                seg0 = AudioSegment.from_file(p)
                seg = ensure_mono16k(seg0)
                seg = pd_effects.normalize(seg)
                td = Path(tempfile.mkdtemp()); tmp_wav = td / (p.stem + "_norm.wav")
                seg.export(tmp_wav, format="wav", parameters=["-acodec","pcm_s16le"])
                audio_for_asr = tmp_wav
                tmp_norm_path = tmp_wav

            if engine == "faster":
                text = transcribe_faster(
                    model, audio_for_asr,
                    language=args_dict["whisper_lang"],
                    beam_size=args_dict["beam_size"],
                    temperature=args_dict["temperature"],
                    vad=args_dict["vad"]
                )
            else:
                text = transcribe_openai(
                    model, audio_for_asr,
                    args_dict["whisper_lang"],
                    args_dict["beam_size"],
                    args_dict["temperature"],
                    args_dict["nospeech_thresh"],
                    args_dict["avglogprob_thresh"],
                    args_dict["comp_ratio_max"]
                )
            text_orth = nfc(text)

            if tmp_norm_path and tmp_norm_path.exists():
                try:
                    tmp_norm_path.unlink(); tmp_norm_path.parent.rmdir()
                except Exception:
                    pass

            # === NEW ORDER: ko_phonology → verbalize ===
            orth_out, text_ref, notes = _pipeline_surface_then_verbalize(text_orth, args_dict)

            # Jamo source
            jamo_src = text_ref if args_dict.get("jamo_source","ref") == "ref" else orth_out
            jamo_ref = jamos_from_text(
                jamo_src,
                keep_space=args_dict.get("jamo_keep_space", False),
                space_token=args_dict.get("jamo_space_token", "▁"),
                keep_ascii=args_dict.get("jamo_keep_ascii", False),
                keep_punct=args_dict.get("jamo_keep_punct", False),
                ascii_lower=args_dict.get("jamo_ascii_lower", False),
            )

            dur_ms = fast_duration_ms(p)
            jamos_out = "".join(jamo_ref) if args_dict.get("jamo_keep_space", False) else " ".join(jamo_ref)
            rows.append({
                "audio": str(p),
                "duration_ms": dur_ms,
                "text": orth_out,
                "text_ref": text_ref,
                "jamos_ref": jamo_ref,
                "jamos_out": jamos_out,
                "notes": notes
            })

            processed += 1
            if args_dict.get("worker_log_every", 0) and (processed % args_dict["worker_log_every"] == 0):
                print(f"[worker@GPU{dev_idx}] processed {processed}/{len(file_list)}")

        except Exception as ex:
            print(f"[WARN][worker@GPU{dev_idx}] {p}: {ex}")
    return rows

# ====================== Main ======================

def parse_gpu_indices(s: Optional[str]) -> List[int]:
    if not s: return []
    try:
        return [int(x) for x in s.split(",") if str(x).strip() != ""]
    except Exception:
        return []

def main():
    os.environ["CT2_FORCE_GPU"] = "1"
    os.environ["CT2_VERBOSE"] = "1"

    ap = argparse.ArgumentParser(description="ASR (Whisper) + ko_phonology + number/unit verbalize (resumable/faster/multiprocess/multi-gpu)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", type=str, help="Folder of audios (recursive)")
    src.add_argument("--from_manifest", type=str, help="CSV manifest with a path column")
    ap.add_argument("--out", required=True, help="Output folder")

    ap.add_argument("--engine", type=str, default="openai", choices=["openai","faster"])
    ap.add_argument("--whisper_model", type=str, default="small")
    ap.add_argument("--whisper_device", type=str, default="auto")
    ap.add_argument("--compute_type", type=str, default="float16")
    ap.add_argument("--whisper_lang", type=str, default="ko")
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--nospeech_thresh", type=float, default=0.6)
    ap.add_argument("--avglogprob_thresh", type=float, default=-1.0)
    ap.add_argument("--comp_ratio_max", type=float, default=2.4)
    ap.add_argument("--vad", action="store_true")

    # Surface + numbers
    ap.add_argument("--emit_surface", action="store_true", help="Apply ko_phonology + extract jamos")
    ap.add_argument("--verbalize_numbers", action="store_true", help="Verbalize digits and ascii units")
    ap.add_argument("--num_reading", type=str, default="auto", choices=["auto","sino","native"])
    ap.add_argument("--verbalize_target", type=str, default="ref", choices=["orth","ref","both"])

    # Audio pre-normalize
    ap.add_argument("--normalize", action="store_true")

    # Resume / partition / logging
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--shard_total", type=int, default=1)
    ap.add_argument("--max_files", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=200)

    # Multi-GPU
    ap.add_argument("--gpu_indices", type=str, default="")
    ap.add_argument("--device_index", type=int, default=-1)

    # Path replace
    ap.add_argument("--path_replace_old", type=str, default=None)
    ap.add_argument("--path_replace_new", type=str, default=None)

    # Jamo options
    ap.add_argument("--jamo_keep_space", action="store_true")
    ap.add_argument("--jamo_space_token", type=str, default="▁")
    ap.add_argument("--jamo_keep_ascii", action="store_true")
    ap.add_argument("--jamo_keep_punct", action="store_true")
    ap.add_argument("--jamo_ascii_lower", action="store_true")
    ap.add_argument("--jamo_source", type=str, default="ref", choices=["ref","orth"])

    # Multi-GPU chunk & worker progress
    ap.add_argument("--gpu_chunk_size", type=int, default=500)
    ap.add_argument("--worker_log_every", type=int, default=500)

    ap.add_argument("--fw_batch_size", type=int, default=16)
    ap.add_argument("--fw_chunk_len_s", type=float, default=30.0)
    ap.add_argument("--fw_num_workers", type=int, default=1)
    ap.add_argument("--fw_cpu_threads", type=int, default=0)

    args = ap.parse_args()

    OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "slices_asr.csv"
    jsonl_path = OUT / "slices_asr.jsonl"

    if args.from_manifest:
        files = rows_from_manifest(Path(args.from_manifest), args.path_replace_old, args.path_replace_new)
    else:
        files = find_audio_files(Path(args.root))
    if not files:
        print("No audio files found."); return

    files = partition_paths(files, args.shard_idx, args.shard_total)

    done_set: Set[str] = set()
    if args.resume:
        done_set = load_done_set(csv_path, jsonl_path)
        if done_set:
            print(f"[RESUME] Loaded {len(done_set):,} completed entries. Skipping them in this run.")
            before = len(files)
            files = [p for p in files if str(p) not in done_set]
            print(f"[RESUME] {before:,} → {len(files):,} remaining.")

    if args.max_files and args.max_files > 0:
        files = files[:args.max_files]
        print(f"[INFO] Limiting to first {args.max_files} files for this run.")

    if not files:
        print("Nothing to do (all files already processed?)."); return

    print(f"[ASR] engine={args.engine}, model={args.whisper_model}, device={args.whisper_device}, files={len(files):,}")

    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["audio","duration_ms","text","text_ref","jamos_out","notes"])

    def write_rows(rows: List[Dict[str, Any]]):
        if not rows: return
        with jsonl_path.open("a", encoding="utf-8") as jf, csv_path.open("a", newline="", encoding="utf-8") as cf:
            w = csv.writer(cf)
            for r in rows:
                jf.write(json.dumps(r, ensure_ascii=False) + "\n")
                w.writerow([r["audio"], r["duration_ms"], r["text"], r["text_ref"],
                            r.get("jamos_out",""), "; ".join(r["notes"] or [])])

    total_written = 0

    gpu_list = parse_gpu_indices(args.gpu_indices)
    if gpu_list:
        def chunked(seq, n):
            for i in range(0, len(seq), n):
                yield seq[i:i+n]

        per_gpu_lists = [files[i::len(gpu_list)] for i in range(len(gpu_list))]
        n_chunk = max(1, args.gpu_chunk_size)

        chunks_by_gpu = []
        for gidx, gpu_files in zip(gpu_list, per_gpu_lists):
            sub_count = math.ceil(len(gpu_files) / n_chunk)
            print(f"[LAUNCH] GPU#{gidx}: {len(gpu_files):,} files in {sub_count} sub-chunks")
            chunks_by_gpu.append(list(chunked(gpu_files, n_chunk)))

        mp_ctx = mp.get_context("spawn")
        executors = {gidx: ProcessPoolExecutor(max_workers=1, mp_context=mp_ctx) for gidx in gpu_list}

        base_args = vars(args).copy()
        base_args["workers"] = 0

        futs = []
        more = True; k = 0
        while more:
            more = False
            for gi, gidx in enumerate(gpu_list):
                if k < len(chunks_by_gpu[gi]):
                    sub = chunks_by_gpu[gi][k]
                    ad = base_args.copy()
                    ad["device_index"] = gidx
                    futs.append(executors[gidx].submit(_worker_run, sub, ad))
                    more = True
            k += 1

        for fu in as_completed(futs):
            rows = fu.result()
            write_rows(rows)
            total_written += len(rows)
            print(f"[PROGRESS][multi-gpu] merged {total_written:,} rows")

        for ex in executors.values():
            ex.shutdown(wait=True)

        print(f"DONE. Newly processed entries this run: {total_written}")
        print("CSV  :", csv_path.resolve())
        print("JSONL:", jsonl_path.resolve())
        return

    if args.workers and args.workers > 0:
        chunks = [files[i::args.workers] for i in range(args.workers)]
        args_dict = vars(args)
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_worker_run, ch, args_dict) for ch in chunks if ch]
            for fu in as_completed(futs):
                rows = fu.result()
                write_rows(rows)
                total_written += len(rows)
                print(f"[PROGRESS] merged {total_written:,} rows from workers")
        print(f"DONE. Newly processed entries this run: {total_written}")
        print("CSV  :", csv_path.resolve())
        print("JSONL:", jsonl_path.resolve())
        return

    if args.device_index is not None and args.device_index >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_index)
        print(f"[GPU/single] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    if args.engine == "faster":
        model = load_faster_model(args.whisper_model, args.whisper_device, args.compute_type, device_index=0)
    else:
        model = load_openai_model(args.whisper_model, args.whisper_device, device_index=0)

    def process_one(p: Path) -> Dict[str, Any]:
        tmp_norm_path = None
        audio_for_asr = p
        if args.normalize:
            seg0 = AudioSegment.from_file(p)
            seg = ensure_mono16k(seg0)
            seg = pd_effects.normalize(seg)
            td = Path(tempfile.mkdtemp()); tmp_wav = td / (p.stem + "_norm.wav")
            seg.export(tmp_wav, format="wav", parameters=["-acodec","pcm_s16le"])
            audio_for_asr = tmp_wav
            tmp_norm_path = tmp_wav

        if args.engine == "faster":
            text = transcribe_faster(
                model, audio_for_asr,
                language=args.whisper_lang,
                beam_size=args.beam_size,
                temperature=args.temperature,
                vad=args.vad
            )
        else:
            text = transcribe_openai(
                model, audio_for_asr,
                args.whisper_lang,
                args.beam_size,
                args.temperature,
                args.nospeech_thresh,
                args.avglogprob_thresh,
                args.comp_ratio_max
            )
        text_orth = nfc(text)

        # === NEW ORDER: ko_phonology → verbalize ===
        orth_out, text_ref, notes = _pipeline_surface_then_verbalize(text_orth, vars(args))

        jamo_src = text_ref if args.jamo_source == "ref" else orth_out
        jamo_ref = jamos_from_text(
            jamo_src,
            keep_space=args.jamo_keep_space,
            space_token=args.jamo_space_token,
            keep_ascii=args.jamo_keep_ascii,
            keep_punct=args.jamo_keep_punct,
            ascii_lower=args.jamo_ascii_lower,
        )

        dur_ms = fast_duration_ms(p)
        jamos_out = "".join(jamo_ref) if args.jamo_keep_space else " ".join(jamo_ref)
        return {
            "audio": str(p),
            "duration_ms": dur_ms,
            "text": orth_out,
            "text_ref": text_ref,
            "jamos_ref": jamo_ref,
            "jamos_out": jamos_out,
            "notes": notes
        }

    total_written = 0
    acc_rows: List[Dict[str, Any]] = []
    for i, p in enumerate(tqdm(files, desc="ASR", unit="file"), 1):
        try:
            row = process_one(p)
            acc_rows.append(row)
            if args.log_every and (i % args.log_every == 0):
                write_rows(acc_rows); total_written += len(acc_rows); acc_rows = []
                print(f"[PROGRESS] written {i:,}/{len(files):,}")
        except Exception as ex:
            print(f"[WARN] {p}: {ex}")

    if acc_rows:
        write_rows(acc_rows); total_written += len(acc_rows)

    print(f"DONE. Newly processed entries this run: {total_written}")
    print("CSV  :", csv_path.resolve())
    print("JSONL:", jsonl_path.resolve())

if __name__ == "__main__":
    main()
