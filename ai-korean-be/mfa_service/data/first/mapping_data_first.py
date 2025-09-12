import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
mapping_path = BASE_DIR / "mapping.json"

def get_paths_by_text(query_text):
    if not mapping_path.exists():
        print(f"❌ File mapping không tồn tại: {mapping_path}")
        return None
    
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping_list = json.load(f)
    
    # Tìm trong list mapping entry có text giống query_text
    for entry in mapping_list:
        if entry.get("text") == query_text:
            # Trả về path wav và textgrid (đã là string)
            return BASE_DIR / entry.get("wav"), BASE_DIR / entry.get("textgrid")
    
    print("⚠️ Không tìm thấy dữ liệu phù hợp với text đã cho.")
    return None