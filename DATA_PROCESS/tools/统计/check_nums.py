import os
import json
count = 0

# 异常样本

with open("/nas_data/taoss/Multi_model_label/labels/multi_model_label.json", "r", encoding="utf-8") as f:
    old_data = json.load(f)
print(f"有缺陷有图片：{len(old_data)}")
count = count + len(old_data)
shengchan = 0
shiyan=0
for item in old_data:
    image = item.get("images")[0]

    if "Image" in image or "铺粉" in image or "打印" in image:
        shengchan = shengchan + 1
    else:
        shiyan = shiyan + 1
print(f"生产环境对话数量：{shengchan}")
print(f"实验环境对话数量：{shiyan}")

with open("/nas_data/taoss/Multi_model_label/labels/multi_model_label_category.json", "r", encoding="utf-8") as f:
    old_data = json.load(f)
    print(len(old_data))
# 正常样本

with open("/nas_data/taoss/Multi_model_label/labels/normal.json", "r", encoding="utf-8") as f:
    old_data = json.load(f)
    print("没有缺陷有图片：",len(old_data))
    count = count + len(old_data)


with open("/nas_data/taoss/Multi_model_label/labels/normal_category.json", "r", encoding="utf-8") as f:
    old_data = json.load(f)
    print(len(old_data))


with open("/nas_data/taoss/Multi_model_label/labels/withoutImage.json", "r", encoding="utf-8") as f:
    old_data = json.load(f)
    print("没有缺陷没有图片通用知识：",len(old_data))
    count = count + len(old_data)

with open("/nas_data/taoss/Multi_model_label/labels/withoutImage_category.json", "r", encoding="utf-8") as f:
    old_data = json.load(f)
    print(len(old_data))


with open("/nas_data/taoss/Multi_model_label/labels/corrective.json", "r", encoding="utf-8") as f:
    old_data = json.load(f)
    print("纠错能力样本：",len(old_data))
    count = count + len(old_data)


with open("/nas_data/taoss/Multi_model_label/labels/corrective_category.json", "r", encoding="utf-8") as f:
    old_data = json.load(f)
    print(len(old_data))

print("总样本数量",count)