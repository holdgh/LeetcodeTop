import os
import json
count = 0
# with open("/nas_data/taoss/Multi_model_label/labels/multi_model_label.json", "r", encoding="utf-8") as f:
#     old_data = json.load(f)
# with open("/nas_data/taoss/Multi_model_label/chaifen/multi_model_label1.json", "w", encoding="utf-8") as f:
#     json.dump(old_data[:5000], f, ensure_ascii=False, indent=4)
# with open("/nas_data/taoss/Multi_model_label/chaifen/multi_model_label2.json", "w", encoding="utf-8") as f:
#     json.dump(old_data[5000:7000], f, ensure_ascii=False, indent=4)
# with open("/nas_data/taoss/Multi_model_label/chaifen/multi_model_label3.json", "w", encoding="utf-8") as f:
#     json.dump(old_data[7000:9000], f, ensure_ascii=False, indent=4)
# with open("/nas_data/taoss/Multi_model_label/chaifen/multi_model_label4.json", "w", encoding="utf-8") as f:
#     json.dump(old_data[9000:], f, ensure_ascii=False, indent=4)


# with open("/nas_data/taoss/Multi_model_label/chaifen/multi_model_label1.json", "r", encoding="utf-8") as f:
#     old_data = json.load(f)
#     count = count + len(old_data)

# with open("/nas_data/taoss/Multi_model_label/chaifen/multi_model_label2.json", "r", encoding="utf-8") as f:
#     old_data = json.load(f)
#     count = count + len(old_data)

# with open("/nas_data/taoss/Multi_model_label/chaifen/multi_model_label3.json", "r", encoding="utf-8") as f:
#     old_data = json.load(f)
#     count = count + len(old_data)

# with open("/nas_data/taoss/Multi_model_label/chaifen/multi_model_label4.json", "r", encoding="utf-8") as f:
#     old_data = json.load(f)
#     count = count + len(old_data)

# print(count)
all = []
with open("/nas_data/taoss/Multi_model_label/new/multi_model_label.json", "r", encoding="utf-8") as f:
    old_data = json.load(f)
    count = count + len(old_data)
all = [item for item in old_data]

with open("/nas_data/taoss/Multi_model_label/new/multi_model_label4.json", "r", encoding="utf-8") as f:
    old_data2 = json.load(f)
    count = count + len(old_data2[-628:])
for item in old_data2[-628:]:
    all.append(item)
print(count)
print(len(all))

with open("/nas_data/taoss/Multi_model_label/new/multi_model_label_all.json", "w", encoding="utf-8") as f:
    json.dump(all, f, ensure_ascii=False, indent=4)