import json
from ClassifyTheProblem import classify_question
count =0 

with open("/nas_data/taoss/Multi_model_label/new/corrective.json", "r", encoding="utf-8") as f:
    corrective = json.load(f)
    print("纠错能力样本：",len(corrective))

with open("/nas_data/taoss/Multi_model_label/labels/corrective_category.json", "r", encoding="utf-8") as f:
    category = json.load(f)
    print(len(category))

corrective_image = [item["images"][0] for item in corrective]
category_image = [item["images"][0] for item in category]
print(len(corrective_image))
print(len(category_image))

diff = list(set(corrective_image) - set(category_image))

for path in diff:
    for item in corrective:
        image = item["images"][0]
        if path == image:
            message = item["messages"]
            for chat in message:
                if chat["role"] == "user":
                    content = chat.get("content", "")
                    classs = classify_question(content)
                    chat["category"] = classs

            category.append(item)
        
with open("/nas_data/taoss/Multi_model_label/labels/corrective_category111.json", "w", encoding="utf-8") as f:
    json.dump(category, f, ensure_ascii=False, indent=4)

                

            




# with open("/nas_data/taoss/Multi_model_label/labels/corrective_category_刮刀横竖条纹.json", "r", encoding="utf-8") as f:
#         old_data = json.load(f)
#         print("纠错能力样本：",len(old_data))
# new_data = []
# for item in old_data:
#     message = item.get("messages")
#     image = item.get("images")[0]
#     for chat in message:
#         if chat["role"] == "system" and "刮刀横/竖条纹" in chat["content"]:
#                 count = count +1


# for item in old_data:
#     message = item.get("messages")
#     image = item.get("images")[0]
#     for chat in message:
#         if chat["role"] == "system" and "刮刀横/竖条纹" not in chat["content"]:
#                 new_data.append(item)
# print(len(new_data))


# with open("../labels/174.json", "w", encoding="utf-8") as f:
#     json.dump(new_data, f, ensure_ascii=False)


# print("刮刀条纹异常：",count)
