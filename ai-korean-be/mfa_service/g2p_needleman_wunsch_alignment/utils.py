def load_mfa_dict(dict_path):
    mfa_dict = {}
    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == "" or line.startswith("#"):
                continue
            parts = line.strip().split()
            key = parts[0]
            phonemes = parts[5:]  # chỉ lấy các phoneme thực sự
            mfa_dict[key] = phonemes
    return mfa_dict

def text_to_phonemes_mfa(text, mfa_dict):
    phonemes = []
    i = 0
    while i < len(text):
        # ưu tiên match 2 ký tự trước nếu có
        if i+1 < len(text) and text[i:i+2] in mfa_dict:
            phonemes.extend(mfa_dict[text[i:i+2]])
            i += 2
        elif text[i] in mfa_dict:
            phonemes.extend(mfa_dict[text[i]])
            i += 1
        else:
            phonemes.append('?')
            i += 1
    return phonemes