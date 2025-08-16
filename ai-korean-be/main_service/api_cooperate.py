from flask import Blueprint, jsonify, request
from matplotlib.pylab import annotations
import requests
import os, uuid, json
import uuid
from io import BytesIO
from werkzeug.datastructures import FileStorage
from datetime import datetime
import shutil

cooperate_bp = Blueprint('cooperate', __name__)


# BASE_DIR = "ai-korean-be/main_service/cooperate"
# USER_DIR = os.path.join(BASE_DIR, "user")
# ANNOTATOR_DIR = os.path.join(BASE_DIR, "annotator")

@cooperate_bp.route('/api/cooperate_annotator', methods=['POST'])
def cooperate_annotator_save():
    data = request.get_json()
    print(f"[INFO] Nhận dữ liệu từ annotator: {data}")

    try:
        annotator_name = data.get("annotator_name")
        annotator_phone = data.get("annotator_phone")
        annotations = data.get("annotations", [])

        # --- Lưu user info ---
        # Tính số annotation lần này
        new_count = sum(len(item.get("annotations", [])) for item in annotations)
        user_file = os.path.join("cooperate", "user", f"{annotator_phone}.json")
        if os.path.exists(user_file):
            with open(user_file, "r", encoding="utf-8") as f:
                old_data = json.load(f)
            total_count = old_data.get("annotation_count", 0) + new_count
        else:
            total_count = new_count

        user_info = {
            "name": annotator_name,
            "phone": annotator_phone,
            "annotation_count": total_count,
            "last_submit": datetime.now().isoformat()
        }
        # Cập nhật số lượng annotation
        with open(user_file, "w", encoding="utf-8") as f:
            json.dump(user_info, f, ensure_ascii=False, indent=2)

        # --- Lưu annotations chi tiết ---
        for ann in annotations:
            audio_path = ann.get("audio_path")

            try:
                filename = os.path.basename(audio_path)
                ann_file = os.path.join("cooperate", "annotator", f"{filename}.json")
                with open(ann_file, "w", encoding="utf-8") as f:
                    json.dump(ann, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[ERROR] Không lưu được {ann_file}: {e}")
                break

            # --- Move audio sang folder cooperate/marked/pronun ---
            if audio_path and os.path.exists(audio_path):
                dest_folder = os.path.join("cooperate", "marked", "pronun")
                os.makedirs(dest_folder, exist_ok=True)

                # Giữ nguyên tên file gốc
                dest_path = os.path.join(dest_folder, filename)
                try:
                    shutil.move(audio_path, dest_path)   # di chuyển file
                    print(f"[INFO] Đã move {audio_path} -> {dest_path}")
                except Exception as e:
                    print(f"[ERROR] Không move được {audio_path}: {e}")
                    continue
                
        return jsonify({"message": "Lưu thành công!"}), 200
    except Exception as e:
        return jsonify({"error": f"Lỗi xử lý: {e}"}), 500

@cooperate_bp.route('/api/cooperate_annotator', methods=['GET'])
def cooperate_annotator_get():
    # Lấy offset, limit từ query (mặc định 0, 10)
    offset = int(request.args.get("offset", 0))
    limit = int(request.args.get("limit", 10))
    results = []
    for root, _, files in os.walk("static", "uploads", "pronun"):
            for file in files:
                if file.endswith(".wav"):
                    # Ví dụ: 223451_가.wav
                    parts = file.split("_", 1)
                    if len(parts) == 2:
                        text = os.path.splitext(parts[1])[0]  # bỏ .wav
                    else:
                        text = ""  # nếu không đúng format thì để rỗng

                    # Path relative từ static
                    rel_path = os.path.relpath(os.path.join(root, file), "")
                    audio_path = f"/{rel_path.replace(os.sep, '/')}"
                    results.append({
                        "audio_path": audio_path,
                        "text_clean": text
                    })

        # Lấy slice theo offset, limit
    sliced_files = results[offset:offset+limit]

    return jsonify({
        "data": sliced_files,
        "count": len(results),      # tổng số file có trong hệ thống
        "offset": offset,
        "limit": limit,
        "returned": len(sliced_files) # số file thực sự trả về
    })

@cooperate_bp.route('/api/cooperate_topik_annotator', methods=['POST'])
def cooperate_topik_annotator_save():
    data = request.get_json()
    # print(f"[INFO] Nhận dữ liệu từ cooperate_topik_annotator: {data}")

    try:
        annotator_name = data.get("annotator_name")
        annotator_phone = data.get("annotator_phone")
        annotations = data.get("annotations", [])
        print(f"[INFO] annotation 0",  annotations[0])
        # --- Lưu user info ---
        # Tính số annotation lần này
        new_count = len(annotations)
       
        if new_count > 0:
            user_file = os.path.join("cooperate", "user", f"{annotator_phone}.json")
            if os.path.exists(user_file):
                
                with open(user_file, "r", encoding="utf-8") as f:
                    old_data = json.load(f)
                total_count = old_data.get("annotation_count", 0) + new_count
            else:
                total_count = new_count
            user_info = {
                "name": annotator_name,
                "phone": annotator_phone,
                "annotation_count": total_count,
                "last_submit": datetime.now().isoformat()
            }
        # Cập nhật số lượng annotation
        with open(user_file, "w", encoding="utf-8") as f:
            json.dump(user_info, f, ensure_ascii=False, indent=2)

            # --- Lưu annotations chi tiết ---
            for ann in annotations:
                try:
                    ann_file = os.path.join("cooperate", "topik_annotator", f"{datetime.now().strftime('%Y%m%d%H%M%S')}.json")
                    with open(ann_file, "w", encoding="utf-8") as f:
                        json.dump(ann, f, ensure_ascii=False, indent=2)
                    print(f"[INFO] Đã lưu {ann_file}")
                except Exception as e:
                    print(f"[ERROR] Không lưu được {ann}: {e}")
                    break
                
        return jsonify({"message": "Lưu thành công!"}), 200
    except Exception as e:
        return jsonify({"error": f"Lỗi xử lý: {e}"}), 500