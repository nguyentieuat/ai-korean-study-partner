import os
import requests
import difflib
from grammar_checker.grammar_checker import check_grammar


# Nhập một câu tiếng Hàn có thể sai ngữ pháp
test_sentence = "안녕하새요"

# Gọi hàm kiểm tra
result = check_grammar(test_sentence)

# In kết quả
print("=== KẾT QUẢ KIỂM TRA ===")
print(f"▶ Câu gốc: {result.get('original')}")
print(f"▶ Câu đã sửa: {result.get('corrected')}")
print(f"▶ Highlight: {result.get('highlighted')}")
print(f"▶ Giải thích:\n{result.get('explanation')}")

# Nếu có lỗi
if "error" in result:
    print(f"❌ Lỗi: {result['error']}")
