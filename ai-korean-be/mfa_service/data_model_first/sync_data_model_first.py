import os
import json
import shutil
import subprocess
from pathlib import Path
from pydub import AudioSegment

# Cấu hình đường dẫn
data_folder = Path("../main_service/data")          
audio_base_path = Path("../main_service/")                  
output_folder = Path("data_model_first/model_first")
output_folder.mkdir(parents=True, exist_ok=True)

# Cấu hình đường dẫn đến từ điển và mô hình MFA
dict_path = Path("korean_mfa.dict")
model_path = Path("korean_mfa.zip")
align_output = Path("data_model_first/align_output")
align_output.mkdir(parents=True, exist_ok=True)

# Danh sách mapping sẽ lưu thông tin text + file paths
mapping_list = []

# B1: Convert JSON → .lab + .wav
for json_file in data_folder.glob("*.json"):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        items = data.get("items", [])

        for idx, item in enumerate(items):
            text = item.get("text")
            audio_url = item.get("audioUrl")
            if not text or not audio_url:
                continue

            filename_base = f"{json_file.stem}_{idx}"
            lab_path = output_folder / f"{filename_base}.lab"
            wav_path = output_folder / f"{filename_base}.wav"

            with open(lab_path, "w", encoding="utf-8") as lab_file:
                lab_file.write(text)

            audio_rel_path = audio_url.lstrip("/")
            mp3_path = audio_base_path / Path(audio_rel_path.replace("/", os.sep))

            if not mp3_path.exists():
                print(f"⚠️ File không tồn tại: {mp3_path}")
                continue

            try:
                audio = AudioSegment.from_mp3(mp3_path)
                audio.export(wav_path, format="wav")
            except Exception as e:
                print(f"❌ Lỗi chuyển đổi {mp3_path}: {e}")
                continue

            # Thêm vào mapping list
            mapping_list.append({
                "text": text,
                "wav": str(Path("model_first") / f"{filename_base}.wav"),
                "textgrid": str(Path("align_output") / f"{filename_base}.TextGrid")  # path sẽ có sau bước align
            })

# B2: Chạy MFA align nếu có ít nhất 1 file .lab và .wav
if any(f.suffix == ".lab" for f in output_folder.iterdir()):
    print("🔁 Đang chạy MFA align...")
    try:
        subprocess.run([
            "mfa", "align",
            str(output_folder),  # thư mục chứa .lab và .wav
            str(dict_path),
            str(model_path),
            str(align_output),
            "--clean",
            "--overwrite"
        ], check=True)
        print("✅ Hoàn tất align với MFA.")

        # Sau khi align xong, ghi mapping file ra đĩa
        mapping_path = Path("data_model_first") / "mapping.json"
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(mapping_list, f, ensure_ascii=False, indent=2)
        print(f"📄 Đã lưu file mapping: {mapping_path}")

    except subprocess.CalledProcessError as e:
        print(f"❌ MFA align lỗi: {e}")
else:
    print("⚠️ Không tìm thấy file .lab nào để align.")
