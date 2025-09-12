#!/usr/bin/env python3
# gen_tts_dataset.py
# Sinh dataset TTS tiếng Hàn, phủ âm tiết Hangul U+AC00..U+D7A3
# Tạo wav 16kHz mono bằng gTTS -> mp3 -> ffmpeg -> wav
# Tạo .lab transcript và mfa_index.csv (filename, labname, text)
#
# Requires: pip install gTTS numpy tqdm
# Requires: ffmpeg available in PATH

import os
import random
import sys
import time
import csv
from gtts import gTTS
import numpy as np
from tqdm import tqdm
import platform

# ========== CẤU HÌNH ==========
OUTPUT_DIR = "tts_data"
WAV_DIR = OUTPUT_DIR  # files saved as 0001.wav, 0001.lab
SENTENCES_TXT = os.path.join(OUTPUT_DIR, "sentences.txt")
MFA_INDEX = os.path.join(OUTPUT_DIR, "mfa_index.csv")

NUM_SENTENCES_TARGET = 5000        # tổng file muốn sinh (bỏ qua nếu muốn phủ hết syllable trước)
MIN_SYLLABLES_PER_SENT = 3        # mỗi câu ít nhất
MAX_SYLLABLES_PER_SENT = 8        # mỗi câu nhiều nhất
SLEEP_BETWEEN = 0.2               # giây giữa requests TTS (tăng nếu gặp rate-limit)
LANG = "ko"                       # gTTS language code

# Nếu bạn chỉ muốn phủ hết syllable một lần (mà không cố tạo NUM_SENTENCES_TARGET),
# đặt MAKE_UNTIL_COVER = True; script sẽ tạo tối thiểu đủ để cover all syllables.
MAKE_UNTIL_COVER = True

# ========= KHÔNG SỬA PHẦN DƯỚI NẾU CHƯA HIỂU =========

# Unicode range for Hangul syllables
START = 0xAC00
END = 0xD7A3  # inclusive

def all_hangul_syllables():
    return [chr(cp) for cp in range(START, END + 1)]

def chunk_existing_prefix(syllables, n):
    """Tạo câu ngắn dựa trên n syllables (space-separated)"""
    return " ".join(syllables[:n])

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def tts_save_wav(text, out_wav_path):
    """
    1) gTTS save mp3 temporary
    2) ffmpeg convert to wav 16kHz mono
    """
    tmp_mp3 = out_wav_path + ".mp3"
    try:
        tts = gTTS(text=text, lang=LANG)
        tts.save(tmp_mp3)
    except Exception as e:
        print("gTTS error:", e)
        if os.path.exists(tmp_mp3):
            os.remove(tmp_mp3)
        raise

    # ffmpeg convert
    if not os.path.exists(tmp_mp3):
        raise RuntimeError(f"File mp3 tạm không tồn tại: {tmp_mp3}")

    if platform.system() == "Windows":
        cmd = f'ffmpeg -y -i "{tmp_mp3}" -ar 16000 -ac 1 "{out_wav_path}" -loglevel error'
    else:
        cmd = f'ffmpeg -y -i "{tmp_mp3}" -ar 16000 -ac 1 "{out_wav_path}" > /dev/null 2>&1'

    rc = os.system(cmd)
    if rc != 0:
        # try again verbosely for debug
        print("ffmpeg convert failed, running verbose command to show error")
        os.system(f'ffmpeg -y -i "{tmp_mp3}" -ar 16000 -ac 1 "{out_wav_path}"')
        raise RuntimeError("ffmpeg conversion failed")
    # cleanup mp3
    try:
        os.remove(tmp_mp3)
    except:
        pass

def generate_sentences_cover_syllables(syllables, min_len=3, max_len=8):
    """
    Tạo danh sách câu sao cho phủ hết syllable.
    Ý tưởng: mỗi câu chọn 3-8 syllables chưa xuất hiện, lấp đầy bằng các syllables random.
    """
    remaining = set(syllables)
    sentences = []
    random_sylls = syllables.copy()
    random.shuffle(random_sylls)

    # 1) Tạo các câu ưu tiên chứa syllables chưa cover
    while remaining:
        take = random.randint(min_len, max_len)
        chosen = []
        for _ in range(take):
            if remaining:
                s = remaining.pop()
                chosen.append(s)
            else:
                # nếu hết remaining, bổ sung syllable random
                chosen.append(random.choice(syllables))
        # để câu đọc tự nhiên, join bằng space
        sentences.append(" ".join(chosen))
        # safety to avoid huge loop
        if len(sentences) > len(syllables):
            break
    return sentences

def generate_more_random_sentences(base_pool, n_more, min_len, max_len):
    more = []
    for _ in range(n_more):
        k = random.randint(min_len, max_len)
        sent = " ".join(random.choices(base_pool, k=k))
        more.append(sent)
    return more

def main():
    print("Start generating TTS dataset...")
    ensure_dir(OUTPUT_DIR)

    # Build hangul syllables list (this is large: ~11,172 syllables)
    syllables = all_hangul_syllables()
    total_syll = len(syllables)
    print(f"Total hangul syllables: {total_syll}")

    # Generate sentences to cover all syllables
    cover_sentences = generate_sentences_cover_syllables(syllables,
                                                         MIN_SYLLABLES_PER_SENT,
                                                         MAX_SYLLABLES_PER_SENT)
    print(f"Generated {len(cover_sentences)} sentences to cover all syllables.")

    sentences = cover_sentences.copy()

    # If want more to reach NUM_SENTENCES_TARGET
    if not MAKE_UNTIL_COVER and NUM_SENTENCES_TARGET > len(sentences):
        need = NUM_SENTENCES_TARGET - len(sentences)
        more = generate_more_random_sentences(syllables, need, MIN_SYLLABLES_PER_SENT, MAX_SYLLABLES_PER_SENT)
        sentences.extend(more)
    elif MAKE_UNTIL_COVER and NUM_SENTENCES_TARGET > len(sentences):
        # vẫn có option: create additional random sentences up to target
        need = max(0, NUM_SENTENCES_TARGET - len(sentences))
        if need > 0:
            print(f"Adding {need} extra random sentences to reach target {NUM_SENTENCES_TARGET}")
            more = generate_more_random_sentences(syllables, need, MIN_SYLLABLES_PER_SENT, MAX_SYLLABLES_PER_SENT)
            sentences.extend(more)

    # Shuffle to mix
    random.shuffle(sentences)

    # Save sentences.txt
    with open(SENTENCES_TXT, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(s + "\n")
    print(f"Wrote sentences list to {SENTENCES_TXT} ({len(sentences)} sentences)")

    # Generate audio + lab + index csv
    rows = []
    pbar = tqdm(enumerate(sentences, start=1), total=len(sentences))
    for idx, sent in pbar:
        file_id = f"{idx:05d}"
        wav_path = os.path.join(WAV_DIR, f"{file_id}.wav")
        lab_path = os.path.join(WAV_DIR, f"{file_id}.lab")

        pbar.set_description(f"Generating {file_id}")

        # If wav already exists skip
        if os.path.exists(wav_path) and os.path.exists(lab_path):
            rows.append((f"{file_id}.wav", f"{file_id}.lab", sent))
            continue

        # Save transcript .lab
        with open(lab_path, "w", encoding="utf-8") as lf:
            lf.write(sent)

        # TTS -> wav
        try:
            tts_save_wav(sent, wav_path)
        except Exception as e:
            print("Error generating TTS for index", idx, "error:", e)
            # remove lab if wav not created
            try:
                os.remove(lab_path)
            except:
                pass
            # optional: backoff and retry once
            time.sleep(2)
            try:
                tts_save_wav(sent, wav_path)
            except Exception as e2:
                print("Retry failed for", file_id, "skipping.")
                continue

        rows.append((f"{file_id}.wav", f"{file_id}.lab", sent))

        # Throttle requests to avoid rate limits
        time.sleep(SLEEP_BETWEEN)

    # Write mfa_index.csv
    with open(MFA_INDEX, "w", encoding="utf-8", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["wav", "lab", "text"])
        for r in rows:
            writer.writerow(r)

    print("Done. Generated files:", len(rows))
    print("Index CSV:", MFA_INDEX)
    print("You can now use these files with MFA train/align.")

if __name__ == "__main__":
    main()
