import re
from difflib import SequenceMatcher


# 1. 去重函数（文本相似度≥90%判定为重复）
def is_similar(text1, text2, threshold=0.9):
    return SequenceMatcher(None, text1, text2).ratio() >= threshold


# 2. 去噪函数（删乱码、无效符号、超长文本）
def clean_text(text):
    # 删特殊符号/乱码
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9，。！？：；""''（）()、]", "", text)
    # 截断超长文本（指令微调单条文本≤512字符）
    return text[:512] if len(text) > 512 else text


if __name__ == '__main__':
    text1 = "我爱打篮球*&*……&%&"
    text2 = "我喜欢打篮球￥%…………*&（*"
    original_texts = [text1, text2]
    end_result = []
    for text_item in original_texts:
        text_item = clean_text(text_item)
        is_duple = False
        for item in end_result:
            if is_similar(item, text_item):
                is_duple = True
                break
        if not is_duple:
            end_result.append(text_item)
    print(f"去重清洗后的文本列表：{end_result}")
