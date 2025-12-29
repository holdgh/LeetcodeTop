import json
from ClassifyTheProblem import classify_question
count =0 

with open("/nas_data/taoss/Multi_model_label/new/withoutImage.json", "r", encoding="utf-8") as f:
    corrective = json.load(f)
    print("纠错能力样本：",len(corrective))

without_questions = []
for item in corrective:
    messages = item.get("messages",[])
    questions = []
    for chat in messages:
        if chat["role"] =="user":
            questions.append(chat["content"])
    without_questions.append(questions)

with open("/nas_data/taoss/Multi_model_label/labels/withoutImage.json", "r", encoding="utf-8") as f:
    category = json.load(f)
    print(len(category))

category_question = []
for item in category:
    messages = item.get("messages",[])
    questions = []
    for chat in messages:
        if chat["role"] =="user":
            questions.append(chat["content"])
    category_question.append(questions)



# corrective_image = [item["images"][0] for item in corrective]
# category_image = [item["images"][0] for item in category]
print(len(without_questions))
print(len(category_question))


temp_dict = {}
for item in category_question:
    temp_dict[str(item)] = temp_dict.get(str(item),0) +1

count = 0
for key,value in temp_dict.items():
    if value >1:
        count = count+value-1
        print(value,key)
print(count)


# for path in diff:
#     for item in corrective:
#         image = item["images"][0]
#         if path == image:
#             message = item["messages"]
#             for chat in message:
#                 if chat["role"] == "user":
#                     content = chat.get("content", "")
#                     classs = classify_question(content)
#                     chat["category"] = classs

#             category.append(item)
        
# with open("/nas_data/taoss/Multi_model_label/labels/corrective_category111.json", "w", encoding="utf-8") as f:
#     json.dump(category, f, ensure_ascii=False, indent=4)

                

            
