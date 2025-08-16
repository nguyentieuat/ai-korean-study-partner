

from jamo import h2j

# Hàm tách câu thành từ đơn giản (nếu có dấu cách)
def split_sentence_to_words(sentence):
    return sentence.strip().split()

# Đọc sentences.txt
with open("tts_data/sentences.txt", "r", encoding="utf-8") as f:
    sentences = f.readlines()

# Tách và lấy từ duy nhất
words = set()
for sent in sentences:
    for w in split_sentence_to_words(sent):
        words.add(w)

# Chuyển từ sang jamo và ghi lexicon
with open("korean_lexicon.txt", "w", encoding="utf-8") as f_out:
    for w in sorted(words):
        jamos = list(h2j(w))
        f_out.write(w + " " + " ".join(jamos) + "\n")
