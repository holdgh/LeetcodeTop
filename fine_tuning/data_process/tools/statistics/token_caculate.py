import json
import re
from transformers import AutoTokenizer

# 示例

collection = ""
files = ["../labels/corrective.json",
              "../labels/multi_model_label.json",
              "../labels/normal.json",
              "../labels/withoutImage.json"
              ]
for file in files:

    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)

        # 2. 统计分类数量

        for item in data:
            message = item.get("messages")
            for chat in message:
                if chat["role"] != "system":
                    collection = collection + chat.get("content", "")
model_name = "/nas_data/models/Qwen/Qwen3-32B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokens = tokenizer.tokenize(collection)
print(f'token数：{len(tokens)}')
print(f"数据集长度{len(collection)}")
# print(f"清除特殊字符----------------------")
# collection = re.sub(r'[^\u4e00-\u9fa5\s]|\s+', ' ', collection, flags=re.UNICODE)
# tokens = tokenizer.tokenize(collection)
# print(f'token数：{len(tokens)}')
# print(f"数据集长度{len(collection)}")