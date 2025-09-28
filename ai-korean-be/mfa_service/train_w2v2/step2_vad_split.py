# step2_vad_split.py
import argparse, os, pathlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import soundfile as sf
import webrtcvad

SAMPLE_RATE = 16000

def ms2samp(ms): return int(SAMPLE_RATE * ms / 1000)

def vad_segments(
    audio, sr,
    aggressiveness=2,
    frame_ms=10,                 # 10ms mịn hơn 20ms
    min_speech_ms=240,
    min_sil_ms=250,              # tăng nhẹ để tránh chẻ câu
    max_len_ms=12000,
    pad_ms=220,                  # pad rộng hơn 180ms
    min_clip_ms=300,
    max_clip_ms=15000,
    start_trigger_ms=50,         # cần ≥50ms voiced liên tiếp để BẮT ĐẦU
    end_trigger_ms=120           # cần ≥120ms unvoiced liên tiếp để KẾT THÚC
):
    """
    VAD với hysteresis: chỉ mở segment khi đủ voiced liên tiếp, và chỉ đóng khi đủ unvoiced liên tiếp.
    Giảm cắt nhầm biên / đứt phụ âm cuối.
    """
    assert sr == SAMPLE_RATE and audio.ndim == 1
    import webrtcvad, numpy as np

    # 16-bit PCM bytes
    pcm16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()

    vad = webrtcvad.Vad(aggressiveness)

    # chia khung 10ms/20ms/30ms → bytes/frame
    bytes_per_sample = 2  # int16
    frame_bytes = int(SAMPLE_RATE * (frame_ms / 1000.0) * bytes_per_sample)
    n_frames = len(pcm16) // frame_bytes

    # flags giọng nói
    speech_flags = []
    for i in range(n_frames):
        chunk = pcm16[i*frame_bytes : (i+1)*frame_bytes]
        if len(chunk) < frame_bytes: break
        speech_flags.append(vad.is_speech(chunk, SAMPLE_RATE))

    # hysteresis counters
    start_need = max(1, start_trigger_ms // frame_ms)
    end_need   = max(1, end_trigger_ms   // frame_ms)

    segments = []
    in_seg = False
    voiced_run = 0
    unvoiced_run = 0
    seg_start_idx = None

    def idx2ms(idx): return idx * frame_ms

    for i, flag in enumerate(speech_flags):
        if flag:
            voiced_run += 1
            unvoiced_run = 0
        else:
            unvoiced_run += 1
            voiced_run = 0

        if not in_seg:
            # chỉ mở khi đủ voiced liên tiếp
            if voiced_run >= start_need:
                in_seg = True
                # lùi về đầu chuỗi voiced để đỡ mất phụ âm đầu
                seg_start_idx = max(0, i - voiced_run + 1)
        else:
            # đang trong segment, chỉ đóng khi đủ unvoiced liên tiếp
            if unvoiced_run >= end_need:
                seg_end_idx = i - unvoiced_run + 1
                st_ms = idx2ms(seg_start_idx)
                et_ms = idx2ms(seg_end_idx)
                # áp pad
                st_ms = max(0, st_ms - pad_ms)
                et_ms = et_ms + pad_ms
                if et_ms - st_ms >= min_speech_ms:
                    segments.append([st_ms, et_ms])
                in_seg = False
                seg_start_idx = None

    # nếu file kết thúc khi còn trong segment → đóng lại
    if in_seg and seg_start_idx is not None:
        st_ms = idx2ms(seg_start_idx)
        et_ms = idx2ms(len(speech_flags))
        st_ms = max(0, st_ms - pad_ms)
        et_ms = et_ms + pad_ms
        if et_ms - st_ms >= min_speech_ms:
            segments.append([st_ms, et_ms])

    # merge các segment gần nhau (< min_sil_ms)
    merged = []
    for s, e in segments:
        if not merged or s - merged[-1][1] > min_sil_ms:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    # split quá dài
    final = []
    for s, e in merged:
        cur = s
        while e - cur > max_len_ms:
            final.append([cur, cur + max_len_ms])
            cur += max_len_ms
        final.append([cur, e])

    # clamp & filter min/max duration
    res = []
    for s, e in final:
        s = max(0, s)
        e = max(s, e)
        dur = e - s
        if dur >= min_clip_ms and dur <= max_clip_ms:
            res.append([s, e])

    return res

def process_one(
    wav_path_str: str,
    inp_root_str: str,
    out_root_str: str,
    aggr: int,
    frame_ms: int,
    min_speech_ms: int,
    min_sil_ms: int,
    max_len_ms: int,
    pad_ms: int,
    min_clip_ms: int,
    max_clip_ms: int,
    start_trigger_ms: int,
    end_trigger_ms: int,
):
    import pathlib, soundfile as sf, numpy as np
    from pathlib import Path

    wav_path = Path(wav_path_str)
    inp_root = Path(inp_root_str)
    out_root = Path(out_root_str)

    # read audio
    audio, sr = sf.read(str(wav_path))
    assert sr == SAMPLE_RATE, f"Expect 16kHz, got {sr}"
    assert audio.ndim == 1, "Expect mono"

    # VAD -> segments
    segs = vad_segments(
        audio, sr,
        aggressiveness=aggr,
        frame_ms=frame_ms,
        min_speech_ms=min_speech_ms,
        min_sil_ms=min_sil_ms,
        max_len_ms=max_len_ms,
        pad_ms=pad_ms,
        min_clip_ms=min_clip_ms,
        max_clip_ms=max_clip_ms,
        start_trigger_ms=start_trigger_ms,
        end_trigger_ms=end_trigger_ms,
    )

    # mirror dir tree under out_root
    out_dir = out_root / wav_path.parent.relative_to(inp_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = wav_path.stem
    out_list = []
    for idx, (s_ms, e_ms) in enumerate(segs):
        s_i = int(SAMPLE_RATE * s_ms / 1000)
        e_i = int(SAMPLE_RATE * e_ms / 1000)
        clip = audio[s_i:e_i]
        if len(clip) < int(0.3 * SAMPLE_RATE):
            continue
        out_name = f"{base}_{s_ms:06d}-{e_ms:06d}.wav"
        out_path = out_dir / out_name
        sf.write(str(out_path), clip, SAMPLE_RATE)
        out_list.append(str(out_path))
    return out_list


def main():
    import argparse, pathlib
    from concurrent.futures import ProcessPoolExecutor, as_completed

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--aggr", type=int, default=2)
    ap.add_argument("--frame_ms", type=int, default=20)
    ap.add_argument("--min_speech_ms", type=int, default=240)
    ap.add_argument("--min_sil_ms", type=int, default=200)
    ap.add_argument("--max_len_ms", type=int, default=12000)
    ap.add_argument("--pad_ms", type=int, default=180)
    ap.add_argument("--min_clip_ms", type=int, default=400)
    ap.add_argument("--max_clip_ms", type=int, default=15000)
    ap.add_argument("--start_trigger_ms", type=int, default=50)
    ap.add_argument("--end_trigger_ms", type=int, default=120)
    args = ap.parse_args()

    inp_root = pathlib.Path(args.inp).resolve()
    out_root = pathlib.Path(args.out).resolve()
    wavs = [str(p) for p in inp_root.rglob("*.wav")]

    if not wavs:
        print("Không tìm thấy WAV 16k. Hãy chạy step1 trước.")
        return

    print(f"Found {len(wavs)} wav files. Splitting with WebRTC VAD…")

    # prepare flat arg tuples (pickle-friendly on Windows)
    job_args = [
        (
            wav,
            str(inp_root),
            str(out_root),
            args.aggr,
            args.frame_ms,
            args.min_speech_ms,
            args.min_sil_ms,
            args.max_len_ms,
            args.pad_ms,
            args.min_clip_ms,
            args.max_clip_ms,
            args.start_trigger_ms,
            args.end_trigger_ms,
        )
        for wav in wavs
    ]

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_one, *ja) for ja in job_args]
        for i, fut in enumerate(as_completed(futs), 1):
            _ = fut.result()  # raise early if any worker fails
            if i % 50 == 0:
                print(f"Progress: {i}/{len(futs)}")

    print("Done.")

if __name__ == "__main__":
    main()


# python train_w2v2/step2_vad_split.py --in "E:\USE\My_project\source audio\audio train w2v2_16k" --out "E:\USE\My_project\source audio\output_vad" --workers 8 
#   --aggr 2 --frame_ms 20 
#   --min_speech_ms 240 --min_sil_ms 200 
#   --max_len_ms 15000 --pad_ms 180 
#   --min_clip_ms 400 --max_clip_ms 15000