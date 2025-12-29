import tiktoken
import json

from transformers import AutoTokenizer

def num_tokens_with_huggingface(text: str, model_name: str) -> int:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.encode(text, return_tensors="pt")
    return tokens.shape[1]


def num_tokens_from_GPT(text: str, model_name: str) -> int:
    """计算文本的token数量"""
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens

# 示例

collection = ""
files = ["D:\VScode\leran_test\Multi_model_label\labels\multi_model_label_category.json",
         "D:\VScode\leran_test\Multi_model_label\labels\\normal_category.json",
         "D:\VScode\leran_test\Multi_model_label\labels\corrective_category.json"]
for file in files:

    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)

        # 2. 统计分类数量

        for item in data:
            message = item.get("messages")
            for chat in message:
                if chat["role"] != "system":
                    collection = collection + chat.get("content", "")

print(collection)

model = "gpt-4"  # 或 "gpt-3.5-turbo"
print(num_tokens_from_GPT(collection, model))
print(f"数据集长度{len(collection)}")
# print(num_tokens_with_huggingface(collection,"bert-base-chinese"))