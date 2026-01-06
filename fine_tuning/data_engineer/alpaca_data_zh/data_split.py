import json
from sklearn.model_selection import train_test_split
import re
from difflib import SequenceMatcher


# 去重函数（文本相似度≥90%判定为重复）
def is_similar(text1, text2, threshold=0.9):
    return SequenceMatcher(None, text1, text2).ratio() >= threshold


# 去噪函数（删乱码、无效符号、超长文本）
def clean_text(text):
    # 删特殊符号/乱码
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9，。！？：；""''（）()、]", "", text)
    # 截断超长文本（指令微调单条文本≤512字符）
    return text[:512] if len(text) > 512 else text


# 保存为JSONL格式（datasets库适配）
def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


# 清洗验证集和测试集
def clean_val_test_data(item):
    """
        验证集/测试集清洗函数：仅做去噪和格式标准化，禁止去重、筛选
        """
    # 1. 去噪：仅删除空文本和严重乱码
    if not item["instruction"] or not item["output"]:
        return None  # 空文本直接跳过，不参与评估
    # 仅删除非中文字符的乱码（保留正常的标点、英文）
    item["instruction"] = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9，。！？：；""''（）()、\s]", "", item["instruction"])
    item["input"] = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9，。！？：；""''（）()、\s]", "", item["input"])
    item["output"] = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9，。！？：；""''（）()、\s]", "", item["output"])

    # 2. 格式标准化：仅统一与训练集完全相同的格式，不做其他修改
    # （训练集用了### Instruction:\n...的格式，验证集/测试集也必须用同样的格式，但不截断、不修改内容）
    return item


if __name__ == '__main__':
    # 1. 加载原始数据集（5万条）
    with open("../../alpaca_zh/alpaca_data_zh_51k.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # 2. 截取1000条子集（适配笔记本）
    sample_data = raw_data[:1000]

    # 3. 按7:2:1划分训练集/验证集/测试集
    train_data, rest_data = train_test_split(sample_data, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(rest_data, test_size=1 / 3, random_state=42)
    # 保存划分好的数据集
    # save_jsonl(train_data, "../data/train_data.jsonl")
    # save_jsonl(val_data, "../data/val_data.jsonl")
    # save_jsonl(test_data, "../data/test_data.jsonl")

    # 3. 清洗训练集
    cleaned_train = []
    seen_texts = []
    for item in train_data:
        # 清洗指令/输入/输出
        item["instruction"] = clean_text(item["instruction"])
        item["input"] = clean_text(item["input"])
        item["output"] = clean_text(item["output"])

        # 跳过空文本
        if not item["instruction"] or not item["output"]:
            continue

        # 去重
        combined_text = item["instruction"] + item["input"] + item["output"]
        is_dup = False
        for seen in seen_texts:
            if is_similar(combined_text, seen):
                is_dup = True
                break
        if not is_dup:
            seen_texts.append(combined_text)
            cleaned_train.append(item)

    # 保存清洗后的数据
    save_jsonl(cleaned_train, "../../data/train_data_cleaned.jsonl")
    print(f"清洗后训练集样本数：{len(cleaned_train)}（原{len(train_data)}）")

    # 清洗验证集
    cleaned_val = [item for item in val_data if clean_val_test_data(item) is not None]
    save_jsonl(cleaned_val, "../../data/val_data_cleaned.jsonl")
    print(f"清洗后验证集样本数：{len(cleaned_val)}（原{len(val_data)}）")
    # 清洗测试集
    cleaned_test = [item for item in test_data if clean_val_test_data(item) is not None]
    save_jsonl(cleaned_test, "../../data/test_data_cleaned.jsonl")
    print(f"清洗后测试集样本数：{len(cleaned_test)}（原{len(test_data)}）")

    # 绝对禁止：对验证集/测试集做去重、样本筛选、数据增强
