import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

if __name__ == '__main__':
    # 1. 加载示例数据集（替换为你的本地数据集路径）
    # 示例数据集格式：csv文件，包含text（评论文本）、label（0/1）两列
    df = pd.read_csv("../../jd_comment/dev.csv", encoding="utf-8")
    df = df.drop(labels='dataset', axis=1)  # 仅保留文本及类别标签列
    df = df.dropna()  # 删除缺省值的行
    # 2. 数据清洗（轻量去噪，适配分类任务）
    import re


    def clean_text(text):
        # 删特殊符号/乱码，保留中文
        text = re.sub(r"[^\u4e00-\u9fa5\s]", "", text)
        return text.strip()


    df["sentence"] = df["sentence"].apply(clean_text)
    df = df[df["sentence"].str.len() > 5]  # 保留有效文本（长度＞5）

    # 3. 划分训练集/验证集/测试集（7:2:1）
    train_df, rest_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label"])
    val_df, test_df = train_test_split(rest_df, test_size=1 / 3, random_state=42, stratify=rest_df["label"])

    # 4. 转换为Hugging Face Dataset格式
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
