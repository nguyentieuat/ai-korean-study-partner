import os, sys, torch
from transformers import pipeline

try:
    from uroman import Uroman
    URO = Uroman()
except Exception as e:
    print("⚠️  Chưa có uroman hoặc lỗi import:", e)
    print("👉 Chạy: pip install -U uroman  (yêu cầu Python >= 3.10)")
    sys.exit(2)

def romanize_kor(text: str) -> str:
    return URO.romanize_string(text)

print("HF_HOME:", os.environ.get("HF_HOME"))
device = 0 if torch.cuda.is_available() else -1

for model in ["facebook/mms-tts-kor", "facebook/mms-tts-vie"]:
    print("Preloading:", model)
    pipe = pipeline("text-to-speech", model=model, device=device)
    text = romanize_kor("안녕하세요.") if "kor" in model else "Xin chào."
    out = pipe(text)  # tạo cache và kiểm tra end-to-end
    print("OK:", model, "sr=", out["sampling_rate"], "len=", len(out["audio"]))
print("Done.")
