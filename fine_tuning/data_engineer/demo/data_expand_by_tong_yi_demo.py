import synonyms  # 中文同义词库（需安装：pip install synonyms）

def augment_text(text):
    # 同义词替换（保留核心语义）
    words = text.split()
    augmented_words = []
    for word in words:
        if len(word) > 1:
            syns = synonyms.nearby(word)[0]
            if syns:
                augmented_words.append(syns[0])
            else:
                augmented_words.append(word)
        else:
            augmented_words.append(word)
    return " ".join(augmented_words)


if __name__ == '__main__':
    text1 = "我爱打篮球"
    print(augment_text(text1))