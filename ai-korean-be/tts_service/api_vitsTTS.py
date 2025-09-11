# app.py — Flask TTS service (Edge-TTS backend: m/f, cache, slow_factor, pause)
from flask import Flask, Blueprint, request, jsonify, send_file
from flask_cors import CORS
import os

# import từ engine bạn đã tạo ở bước trước
from tts_engine_edge import synthesize, MODEL_KOR, CACHE_DIR, DEVICE_ID, SAMPLE_RATE, VOICE_MAP

vitsTTS_bp = Blueprint('vitsTTS', __name__)

@vitsTTS_bp.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({
        "ok": True,
        "engine": "edge-tts",
        "model": MODEL_KOR,
        "device": "cuda" if DEVICE_ID >= 0 else "cpu",
        "cache_dir": CACHE_DIR,
        "sample_rate": SAMPLE_RATE,
        "voices": VOICE_MAP,  # {"m": "...InJoon...", "f": "...SunHi..."}
        "env": {
            "EDGE_VOICE_M": os.getenv("EDGE_VOICE_M"),
            "EDGE_VOICE_F": os.getenv("EDGE_VOICE_F"),
        }
    })

@vitsTTS_bp.route("/api/voices", methods=["GET"])
def voices():
    # Trả về danh sách voice hiện map cho m/f
    return jsonify({"ok": True, "voices": VOICE_MAP})


@vitsTTS_bp.route("/api/tts", methods=["POST"])
def tts():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    slow = float(data.get("slow_factor", 0.9))   # 0.5..1.5  (map -> rate %+)
    pause = int(data.get("pause_ms", 400))       # break giữa câu
    use_cache = bool(data.get("cache", True))
    speaker = (data.get("speaker") or "f").lower()  # "m" | "f"

    if not text:
        return jsonify({"ok": False, "error": "text rỗng."}), 400

    try:
        wav_bytes, sr, key = synthesize(
            text=text,
            slow_factor=slow,
            pause_ms=pause,
            use_cache=use_cache,
            speaker=speaker,
        )
        return jsonify({
            "ok": True,
            "sampling_rate": sr,
            "cache_key": key,
            "bytes": len(wav_bytes),
            "speaker": speaker
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@vitsTTS_bp.route("/api/tts/file", methods=["GET"])
def tts_file():
    key = request.args.get("key", type=str)
    filename = request.args.get("filename", default="tts.wav", type=str)
    if not key:
        return jsonify({"ok": False, "error": "missing key"}), 400

    path = os.path.join(CACHE_DIR, f"{key}.wav")
    if not os.path.exists(path):
        return jsonify({"ok": False, "error": "not found"}), 404

    return send_file(path, mimetype="audio/wav", as_attachment=True, download_name=filename)

