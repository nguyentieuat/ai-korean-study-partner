from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torchaudio
import torch
import numpy as np
from textgrid import TextGrid
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# Load model và processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()


def extract_features(wav_path):
    waveform, sr = torchaudio.load(wav_path)

    # Resample nếu cần
    target_sr = 16000
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        sr = target_sr

    input_values = processor(waveform.squeeze(), sampling_rate=sr, return_tensors="pt").input_values
    with torch.no_grad():
        outputs = model(input_values)
    features = outputs.last_hidden_state.squeeze(0)
    total_audio_sec = waveform.shape[1] / sr
    return features, sr, total_audio_sec

def load_alignment(textgrid_path):
    tg = TextGrid.fromFile(textgrid_path)
    phones = []
    for interval in tg[1]:  # Giả sử tier thứ 2 là phoneme
        if interval.mark.strip():
            phones.append({
                "phoneme": interval.mark,
                "start": interval.minTime,
                "end": interval.maxTime
            })
    return phones


def extract_phoneme_vectors(features, sr, phoneme_times, total_audio_sec):
    """Map thời gian phoneme → frame chỉ số"""
    phoneme_vectors = []
    num_frames = features.shape[0]
    for p in phoneme_times:
        start_idx = int(p["start"] / total_audio_sec * num_frames)
        end_idx = int(p["end"] / total_audio_sec * num_frames)
        vec = features[start_idx:end_idx].mean(dim=0)
        phoneme_vectors.append({
            "phoneme": p["phoneme"],
            "vector": vec
        })
    return phoneme_vectors


def cosine_score(vec1, vec2):
    return cosine_similarity([vec1.numpy()], [vec2.numpy()])[0][0]


def score_pronunciation(vectors_user, vectors_ref, phones_user):
    scores = []
    for p_user, p_ref in zip(vectors_user, vectors_ref):
        if p_user['phoneme'] == p_ref['phoneme']:
            sim = cosine_score(p_user['vector'], p_ref['vector'])
            sim_rounded = round(float(sim), 4)

            # Phân loại màu theo similarity
            if sim_rounded >= 0.8:
                color = "green"  # phát âm chuẩn
            elif 0.6 <= sim_rounded < 0.8:
                color = "yellow"  # gần đúng
            else:
                color = "red"  # sai nhiều

            # Lấy start, end từ phones_user tương ứng phoneme
            phone_time = next((p for p in phones_user if p["phoneme"] == p_user['phoneme']), None)

            scores.append({
                "phoneme": p_user['phoneme'],
                "similarity": sim_rounded,
                "start": phone_time["start"] if phone_time else None,
                "end": phone_time["end"] if phone_time else None,
                "color": color
            })

    avg_score = round(np.mean([s["similarity"] for s in scores]) * 100, 2) if scores else 0.0

    return {
        "avg_score": avg_score,
        "details": scores
    }
