import json
# import matplotlib.pyplot as plt
from collections import defaultdict
import os

category_counts = {}
# 解决中文显示问题
# plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统使用黑体
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

files = ["/nas_data/taoss/Multi_model_label/labels/multi_model_label_category.json",
         "/nas_data/taoss/Multi_model_label/labels/normal_category.json",
         "/nas_data/taoss/Multi_model_label/labels/corrective_category.json",
         "/nas_data/taoss/Multi_model_label/labels/withoutImage_category.json"]

dict_for = {}
for file in files:
    temp = {}
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if file == "/nas_data/taoss/Multi_model_label/labels/multi_model_label_category.json":
        shengchan = {}
        shiyan = {}
        for item in data:
            message = item.get("messages")
            image = item.get("images")[0]
            for chat in message:
                if chat["role"] == "user":
                    category = chat.get("category",None)
                    category_counts[category] = category_counts.get(category, 0) + 1
                    temp[category] = temp.get(category, 0) + 1
                    if "Image" in image or "铺粉" in image or "打印" in image:
                        shengchan[category] = shengchan.get(category, 0) + 1
                    else:
                        shiyan[category] = shiyan.get(category, 0) + 1
        dict_for[file] = temp
    else:
        for item in data:
            message = item.get("messages") 
            for chat in message:
                if chat["role"] == "user":
                    category = chat.get("category",None)
                    category_counts[category] = category_counts.get(category, 0) + 1
                    temp[category] = temp.get(category, 0) + 1
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
print("生产环境问题总数:", shengchan_total)
shiyan_total = sum(shiyan.values())
print("实验环境问题总数:", shiyan_total)


# import matplotlib.pyplot as plt
# import seaborn as sns

# # ===== 关键修正点 =====
# sns.set_theme(style="whitegrid")  # 替换旧的plt.style.use('seaborn')
# # ======================

# # 设置中文字体
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False

# # 示例数据
# categories = list(category_counts.keys())
# counts = list(category_counts.values())

# # 创建图表
# fig, ax = plt.subplots(figsize=(8,6))

# # 绘制柱状图（现代风格）
# bars = ax.bar(
#     x=categories,
#     height=counts,
#     color=sns.color_palette("husl", len(categories)),
#     edgecolor='w',
#     width=0.5,  # 关键修改：减小柱状图宽度
#     linewidth=1,
#     alpha=0.8
# )

# # 添加数据标签
# for bar in bars:
#     height = bar.get_height()
#     ax.text(
#         bar.get_x() + bar.get_width()/2., height,
#         f'{height}',
#         ha='center', va='bottom',
#         fontsize=11
#     )

# # 美化图表
# ax.set_title('问题分类统计',
#             fontsize=14, pad=20, fontweight='bold')
# ax.set_xlabel('分类', fontsize=12)
# ax.set_ylabel('数量', fontsize=12)
# plt.xticks(rotation=45, ha='right')
# # 添加网格线
# ax.yaxis.grid(True, linestyle='--', alpha=0.4)
# # 移除顶部和右侧边框
# sns.despine()
# plt.tight_layout()
# plt.show()