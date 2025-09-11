# ==== FIX monotonic_align + SMOKE TEST VITS ====
import os, sys, re, json, glob, pathlib, subprocess, importlib

# --- Locate repo (Kaggle/Colab đều ok) ---
CANDIDATES = [
    "/kaggle/working/vits_kor_ms2/vits",
    "/content/vits_kor_ms2/vits",
    os.getcwd(),                     # fallback: đang ở /content/.../vits
]
REPO_ROOT = None
for c in CANDIDATES:
    p = pathlib.Path(c)
    if (p/"models.py").exists() or (p/"models").exists():
        REPO_ROOT = p; break
assert REPO_ROOT, "Không tìm thấy thư mục repo VITS (chứa models.py). Hãy chỉnh lại CANDIDATES."
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
print("[repo]", REPO_ROOT)

# --- Ensure monotonic_align imports correctly ---
MA = REPO_ROOT / "monotonic_align"
assert MA.exists(), f"Thiếu folder: {MA}"
init_py = MA / "__init__.py"

def has_top_core():
    if (MA/"core.py").exists(): return True
    return any(p.name.startswith("core") and p.suffix == ".so" for p in MA.glob("core*"))
def has_nested_core():
    d = MA/"monotonic_align"
    if not d.exists(): return False
    if (d/"core.py").exists(): return True
    return any(p.name.startswith("core") and p.suffix == ".so" for p in d.glob("core*"))

def patch_init(line):
    txt = init_py.read_text(encoding="utf-8") if init_py.exists() else ""
    if txt.strip() != line.strip():
        init_py.write_text(line, encoding="utf-8")
        print("[patch] __init__.py ->", line.strip())

def build_extension(folder: pathlib.Path):
    print("[build] building extension in", folder)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "pip", "setuptools", "wheel", "pybind11"])
    if (folder/"setup.py").exists():
        subprocess.check_call([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=str(folder))
    else:
        # một số fork đặt setup.py ở MA/, số khác đặt ở MA/monotonic_align/
        nested = folder/"monotonic_align"
        if (nested/"setup.py").exists():
            subprocess.check_call([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=str(nested))

# 1) Quyết định layout & patch import
top, nested = has_top_core(), has_nested_core()
if top:
    patch_init("from .core import maximum_path_c\n")
elif nested:
    patch_init("from .monotonic_align.core import maximum_path_c\n")
else:
    # chưa có core -> build
    patch_init("from .core import maximum_path_c\n")
    try:
        build_extension(MA)
    except Exception as e:
        print("[warn] build failed:", e)

# 2) Re-evaluate & import test
top, nested = has_top_core(), has_nested_core()
try:
    import monotonic_align
    importlib.reload(monotonic_align)
    print("[ok] monotonic_align imported")
except Exception as e:
    print("[retry] import failed:", e)
    # Thử patch theo layout còn lại
    if nested: patch_init("from .monotonic_align.core import maximum_path_c\n")
    elif top:  patch_init("from .core import maximum_path_c\n")
    # thử import lần nữa
    import monotonic_align
    importlib.reload(monotonic_align)
    print("[ok] monotonic_align imported (after patch)")

# --- Smoke test config ---
EXP_NAME  = "vits_kor_ms2"  # đổi nếu exp khác
EXP_DIR   = (REPO_ROOT / "logs" / EXP_NAME).resolve()
CFG_PATH  = EXP_DIR / "config.json"
assert EXP_DIR.exists(), f"Không thấy exp: {EXP_DIR}"
assert CFG_PATH.exists(), f"Không thấy config.json: {CFG_PATH}"

# --- Pick checkpoint ---
def _step(p):
    m = re.search(r'_(\d+)\.pth$', os.path.basename(str(p))); return int(m.group(1)) if m else -1
def pick_ckpt(exp_dir: pathlib.Path):
    g_latest = exp_dir/"G_latest.pth"
    if g_latest.exists(): return str(g_latest), "latest"
    cands = sorted(list(exp_dir.glob("G_*.pth")) + list(exp_dir.glob("g_*.pth")), key=_step)
    assert cands, f"Không tìm thấy G_*.pth trong {exp_dir}"
    return str(cands[-1]), f"step:{_step(cands[-1])}"
CKPT_PATH, CKPT_TAG = pick_ckpt(EXP_DIR)
print(f"[ckpt] {os.path.basename(CKPT_PATH)} ({CKPT_TAG})")

# --- Load hparams ---
try:
    import utils
    hps = utils.get_hparams_from_file(str(CFG_PATH))
except Exception:
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        _raw = json.load(f)
    class Dot(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
    hps = Dot({k: Dot(v) if isinstance(v, dict) else v for k,v in _raw.items()})

# --- Import model/text ---
from models import SynthesizerTrn
from text import text_to_sequence, symbols

# --- Build model (match train) ---
import torch, soundfile as sf
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_symbols = len(symbols)
spec_channels = hps.data.filter_length // 2 + 1
seg_frames = hps.train.segment_size // hps.data.hop_length
n_speakers_cfg = int(getattr(hps.data, "n_speakers", 0))
gin_channels   = int(getattr(hps.model, "gin_channels", 0))

net_g = SynthesizerTrn(
    n_symbols,
    spec_channels,
    seg_frames,
    n_speakers=n_speakers_cfg,
    **(hps.model)
).to(device).eval()

# --- Load weights ---
ckpt = torch.load(CKPT_PATH, map_location="cpu")
state = None
for k in ["model", "generator", "state_dict", "net_g"]:
    if isinstance(ckpt, dict) and k in ckpt:
        state = ckpt[k]; break
state = state or ckpt
missing, unexpected = net_g.load_state_dict(state, strict=False)
print(f"[model] n_speakers={getattr(net_g,'n_speakers',0)}, gin_channels={gin_channels}, "
      f"emb_g? {hasattr(net_g,'emb_g')} | missing={len(missing)}, unexpected={len(unexpected)}")

# --- Helpers ---
def text_to_seq(txt: str):
    cleaners = getattr(hps.data, "text_cleaners", ["korean_cleaners"])
    return text_to_sequence(txt, cleaners)

@torch.no_grad()
def synth(text: str, sid: int = 0, out_path: str = "smoke.wav",
          length_scale: float = 1.10, noise_scale: float = 0.35, noise_scale_w: float = 0.60):
    seq = text_to_seq(text)
    x = torch.LongTensor([seq]).to(device)
    x_len = torch.LongTensor([len(seq)]).to(device)
    use_spk = (getattr(net_g, "n_speakers", 0) > 1) and hasattr(net_g, "emb_g")
    sid_t = torch.LongTensor([sid]).to(device) if use_spk else None

    out = net_g.infer(
        x, x_len,
        sid=sid_t,
        noise_scale=noise_scale,
        length_scale=length_scale,
        noise_scale_w=noise_scale_w
    )
    audio = out[0] if isinstance(out, (list, tuple)) else out
    wav = audio[0,0].float().cpu().numpy() if audio.dim()==3 else audio[0].float().cpu().numpy()
    sf.write(out_path, wav, int(hps.data.sampling_rate), subtype="PCM_16")
    print(f"[ok] Saved {out_path} | sid={('n/a' if sid_t is None else sid)} | {len(wav)/hps.data.sampling_rate:.2f}s")

# --- Smoke run ---
print("\n[smoke] generating...")
if getattr(net_g, "n_speakers", 0) > 1 and hasattr(net_g, "emb_g"):
    synth("안녕하세요. 테스트 중입니다.", sid=0, out_path="smoke_sid0.wav")
    synth("테스트 음성 합성 결과입니다.", sid=1, out_path="smoke_sid1.wav")
else:
    synth("안녕하세요. 테스트 중입니다.", out_path="smoke.wav")

print("✓ Smoke test done.")


