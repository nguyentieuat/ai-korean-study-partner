from gtts import gTTS
import os

# Văn bản tiếng Hàn
text_ko = "안녕하세요. 만나서 반갑습니다."

# Khởi tạo gTTS với ngôn ngữ tiếng Hàn (ko)
tts = gTTS(text=text_ko, lang='ko')

# Lưu file âm thanh
output_path = "output.mp3"
tts.save(output_path)

print(f"Đã lưu file: {output_path}")

# (Tuỳ chọn) Phát file ngay sau khi lưu (chỉ hoạt động trên Windows/macOS/Linux GUI)
os.system(f"start {output_path}")  # Windows
# os.system(f"afplay {output_path}")  # macOS
# os.system(f"xdg-open {output_path}")  # Linux
