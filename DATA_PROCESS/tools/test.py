import json
import matplotlib.pyplot as plt
from collections import defaultdict
import os

category_counts = {}
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
files = ["../labels/multi_model_label_category.json"]

dict_for = {}
test = 0
for file in files:
    temp = {}
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if file == "../labels/multi_model_label_category.json":
        shengchan = {}
        shiyan = {}
        for item in data:
            message = item.get("messages")
            image = item.get("images")[0]
            for chat in message:
                if chat["role"] == "user":
                    category = chat.get("category",None)
                    if category =="异常原因分析":
                        test =test+1
                    if category:
                        category_counts[category] = category_counts.get(category, 0) + 1
                    temp[category] = temp.get(category, 0) + 1
                    if "Image" in image or "铺粉" in image or "打印" in image:
                        shengchan[category] = shengchan.get(category, 0) + 1
                    else:
                        shiyan[category] = shiyan.get(category, 0) + 1
        dict_for[file] = temp

print(category_counts)
total = sum(category_counts.values())
print("问题总数:", total)
for file,value in dict_for.items():
    print(file)
    print(value)
    total = sum(value.values())
    print(f"问题总数:", total)

shengchan_total = sum(shengchan.values())
print(shengchan)
print("生产环境问题总数:", shengchan_total)
shiyan_total = sum(shiyan.values())
print(shiyan)
print("实验环境问题总数:", shiyan_total)
print(test)
