from flask import Blueprint, jsonify, request
import os, uuid, json,time
import uuid
from datetime import datetime
import shutil
import traceback

cooperate_bp = Blueprint('cooperate', __name__)

@cooperate_bp.route('/api/cooperate_annotator', methods=['POST'])
def cooperate_annotator_save():
    data = request.get_json()

    try:
        annotator_name = data.get("annotator_name")
        annotator_phone = data.get("annotator_phone")
        annotations = data.get("annotations", [])

        # --- Lưu user info ---
        # Tính số annotation lần này
        new_count = len(annotations)
        json_path = ""
        if new_count > 0:
            user_folder = os.path.join("cooperate", "user")
            os.makedirs(user_folder, exist_ok=True)  # tạo folder nếu chưa tồn tại

            user_file = os.path.join(user_folder, f"{annotator_phone}.json")
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

                # --- Move audio sang folder cooperate/marked/pronun ---
                if audio_path and os.path.exists(audio_path):
                    dest_folder = os.path.join("cooperate", "marked", "pronun")
                    os.makedirs(dest_folder, exist_ok=True)

                    # Giữ nguyên tên file gốc
                    dest_path = os.path.join(dest_folder, filename)
                    try:
                        shutil.move(audio_path, dest_path)   # di chuyển file
                        ann["audio_path"] = dest_path
                        print(f"[INFO] Đã move {audio_path} -> {dest_path}")
                    except Exception as e:
                        print(f"[ERROR] Không move được {audio_path}: {e}")
                        traceback.print_exc()
                        continue

                try:
                    filename = os.path.basename(ann["audio_path"])
                    ann_file = os.path.join("cooperate", "annotator", f"{filename}.json")
                    with open(ann_file, "w", encoding="utf-8") as f:
                        json.dump(ann, f, ensure_ascii=False, indent=2)
                    json_path = ann_file
                except Exception as e:
                    print(f"[ERROR] Không lưu được {ann_file}: {e}")
                    traceback.print_exc()
                    break

        return jsonify({"message": "Lưu thành công!", "json_file": json_path}), 200
    except Exception as e:
        print(e)
        traceback.print_exc()
        return jsonify({"error": f"Lỗi xử lý: {e}"}), 500

@cooperate_bp.route('/api/cooperate_annotator', methods=['GET'])
def cooperate_annotator_get():
    try:
        # Lấy offset, limit từ query (mặc định 0, 10)
        offset = int(request.args.get("offset", 0))
        limit = int(request.args.get("limit", 10))
        results = []

        folder = os.path.join("static", "uploads", "pronun")
        if os.path.exists(folder):
            for root, _, files in os.walk(folder):
                for file in files:
                    if file.endswith(".wav"):
                        # Ví dụ: 223451_가.wav
                        parts = file.split("_", 1)
                        text = os.path.splitext(parts[1])[0] if len(parts) == 2 else ""
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

        return jsonify({"error": "Không tìm thấy dữ liệu"}), 404

    except Exception as e:
        print(e)
        traceback.print_exc()
        return jsonify({"error": f"Lỗi khi lấy dữ liệu: {e}"}), 500


@cooperate_bp.route('/api/cooperate_topik_annotator', methods=['POST'])
def cooperate_topik_annotator_save():
    data = request.get_json()
    # print(f"[INFO] Nhận dữ liệu từ cooperate_topik_annotator: {data}")

    try:
        annotator_name = data.get("annotator_name")
        annotator_phone = data.get("annotator_phone")
        annotations = data.get("annotations", [])
 
        # --- Lưu user info ---
        # Tính số annotation lần này
        new_count = len(annotations)
        json_path = ""
        if new_count > 0:
            user_folder = os.path.join("cooperate", "user")
            os.makedirs(user_folder, exist_ok=True)  # tạo folder nếu chưa tồn tại

            user_file = os.path.join(user_folder, f"{annotator_phone}.json")
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
            for i,ann in enumerate(annotations):
                try:
                    ann_file = os.path.join("cooperate", "topik_annotator", f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}.json")
                    with open(ann_file, "w", encoding="utf-8") as f:
                        json.dump(ann, f, ensure_ascii=False, indent=2)
                    json_path = ann_file
                except Exception as e:
                    print(f"[ERROR] Không lưu được {ann}: {e}")
                    break
                
        return jsonify({"message": "Lưu thành công!", "json_file": json_path}), 200
    except Exception as e:
        print(e)
        traceback.print_exc()
        return jsonify({"error": f"Lỗi xử lý: {e}"}), 500

@cooperate_bp.route('/api/cooperate_vitspre', methods=['POST'])
def cooperate_vitspre_save():
    try:
        # Lấy dữ liệu text từ FormData
        annotator_name = request.form.get("annotator_name") or ""
        annotator_phone = request.form.get("annotator_phone")
        records = json.loads(request.form.get("records", "[]"))

        # --- Lưu user info ---
        new_count = len(records)
        json_path = ""  # khai báo trước để trả về nếu không có record
        if new_count > 0:
            user_folder = os.path.join("cooperate", "user")
            os.makedirs(user_folder, exist_ok=True)  # tạo folder nếu chưa tồn tại

            user_file = os.path.join(user_folder, f"{annotator_phone}.json")
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
            os.makedirs(os.path.dirname(user_file), exist_ok=True)
            with open(user_file, "w", encoding="utf-8") as f:
                json.dump(user_info, f, ensure_ascii=False, indent=2)

            # Duyệt qua từng file audio trong request.files
            print(f"[INFO] Nhận records", records)
            saved_records = []
            for i, record in enumerate(records):
                print(f"[INFO] Xử lý record {i+1}: {record}")
                file = request.files.get(f"audio_{i}")
                if file:
                    # Đặt tên file duy nhất
                    filename = f"{uuid.uuid4().hex}_{record.get('text_clean', '')}.wav"
                    audio_path_folder = os.path.join("cooperate", "vitspre", "audio")
                    os.makedirs(audio_path_folder, exist_ok=True)
                    audio_path = os.path.join(audio_path_folder, filename)
                    file.save(audio_path)
                    record["audio_path"] = audio_path
                saved_records.append(record)

            # Lưu metadata JSON kèm path thật
            json_path_folder = os.path.join("cooperate", "vitspre")
            os.makedirs(json_path_folder, exist_ok=True)
            json_path = os.path.join(json_path_folder, f"vitspre_{int(time.time())}_{record['text_clean']}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(saved_records, f, ensure_ascii=False, indent=2)

        return jsonify({"message": "Lưu thành công!", "json_file": json_path}), 200

    except Exception as e:
        print(e)
        traceback.print_exc()
        return jsonify({"error": f"Lỗi lưu Vitspre: {e}"}), 500
