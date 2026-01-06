import torch

torch.set_default_dtype(torch.float32)  # CPU训练默认float32
DEVICE = torch.device("cpu")  # 强制CPU（无GPU时）
from transformers import AutoTokenizer

# 1. 加载Tokenizer（all-MiniLM-L6-v2原生支持中文）
tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\gaohu\aiModel\all-MiniLM-L6-v2")


# 2. 预处理函数（分类任务核心：编码文本+标签）
def preprocess_function(examples):
    # 编码文本：max_length=128（适配短文本分类）
    encoding = tokenizer(
        examples["sentence"],
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    # 标签转换为tensor（分类任务需int型）
    labels = torch.tensor(examples["label"], dtype=torch.long).to(DEVICE)
    encoding["labels"] = examples["label"]
    return encoding


# 2. 优化文本编码（添加情感Prompt，提升特征聚焦）
def preprocess_function_enhanced(examples):
    # 新增情感分类Prompt：让模型聚焦情感特征
    texts = [f"判断以下酒店评论的情感倾向：{text} 情感倾向：" for text in examples["sentence"]]
    encoding = tokenizer(
        texts,
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    labels = torch.tensor(examples["label"], dtype=torch.long).to(DEVICE)
    encoding["labels"] = labels
    return encoding
