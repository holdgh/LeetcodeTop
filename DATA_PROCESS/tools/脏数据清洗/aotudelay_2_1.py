from openai import OpenAI
import json
import re
import os
temperature = 0.8



base_url="http://192.168.0.194:6662/v1"
api_key="test"
model = "Qwen3-32B"

def Valid(temp,file):

        image_path = temp.get("images","")
        messages = temp.get("messages",[])
        questions = []
        for chat in messages:
            if chat["role"] =="user":
                questions.append(chat["content"])
        print(questions)    
        print(image_path)    

        if os.path.exists(file):
            print("存在路径")
            with open(file, "r", encoding="utf-8") as f:
                old_data = json.load(f)
            for item in old_data:
                image_path2 = item["images"]
                questions2 = []
                messages2 = item["messages"]
                if image_path == image_path2:
                    for chat2 in messages2:
                        if chat2["role"] =="user":
                            questions2.append(chat2["content"])
                    if questions == questions2:
                        return False
            return True
        else:
            print("不存在路径")
            return True



def response(text):
    client = OpenAI(base_url=base_url, api_key=api_key)
    messages = [
        {
            "role": "system",
            "content": '''
# 任务要求：请对用户的输入进行修正：删除所有关于缺陷在图片上的位置描述（如 “边缘位置”“中部区域”“中间和上侧” 等具体方位表述），其他内容（缺陷类型、特征描述、分析逻辑）保持不变，不新增任何信息，并且保持语义不变，逻辑通顺。
# 要求：
    - 如果多模态大模型的回答中没有位置描述，则返回用户输入的内容，不做修改。
    - 禁止返回分析过程，仅返回修正后的用户输入文本。例如："用户的输入中没有涉及缺陷在图片上的具体位置描述，因此无需修改，直接返回原内容："
    - 修正后的文本放在标签<text></text>中返回

# 示例：
    ## 输入
        图片中一共出现 3 种缺陷问题。
        第一种缺陷类型是曲翘凸起，主要出现在零件的边缘位置。该区域中可以看到零件的边缘轮廓，呈现白色或金属光泽，没有铺上金属粉末。因此可以分析在这些位置发生了曲翘凸起异常。
        第二种缺陷类型是污染物，出现在图片的中部区域。该位置区域中可以看到明显的深灰色或黑色阴影。因此可以分析在这个位置发生了污染物异常。
        第三种缺陷类型是刮刀横 / 竖条纹，贯穿于图片的中间和上侧。该位置区域展现出深黑色的横线或竖线。因此可以分析在这些位置发生了刮刀横 / 竖条纹异常。

    ## 输出：
        <text>
        图片中一共出现 3 种缺陷问题。
        第一种缺陷类型是曲翘凸起。可以看到零件的边缘轮廓，呈现白色或金属光泽，没有铺上金属粉末。因此可以分析发生了曲翘凸起异常。
        第二种缺陷类型是污染物。可以看到明显的深灰色或黑色阴影。因此可以分析发生了污染物异常。
        第三种缺陷类型是刮刀横/竖条纹。展现出深黑色的横线或竖线。因此可以分析发生了刮刀横/竖条纹异常。
        </text>



''',
        },
        {
            "role": "user",
            "content":f"用户的输入：{text}" 

            }
    ]
    response = client.chat.completions.create(
                model=model,
                messages=messages
            )

    event_text = response.choices[0].message.content if response.choices else ""
    response = re.sub(r'<think>.*?</think>', '', event_text, flags=re.DOTALL)
    response = re.findall(r".*<text>(.*)</text>", response, re.DOTALL)[0]
    return response

def main():
    file_names = [
         "/nas_data/taoss/Multi_model_label/labels/corrective.json"
              ]
    all = []

    for file in file_names:
        with open(file, "r", encoding="utf-8") as f:
            old_data = json.load(f)
            all.append([file,old_data]) 
    
 
    print(len(all[0][1]))
    for data in all:
        file = data[0]
        file_new = "/nas_data/taoss/Multi_model_label/new/"+file.split("/")[-1]
        old_data = data[1]
        for item in old_data:
            flag = Valid(item,file_new)
            print(flag)
            if flag:
                message = item.get("messages")
                for chat in message:
                    if chat["role"] == "assistant":
                        content = chat.get("content", None)
                        print(content)
                        print("清理中")
                        new_content = response(content)
                        print(new_content)
                        chat["content"] = new_content



                if os.path.exists(file_new):
                    with open(file_new, "r", encoding="utf-8") as f:
                        temp = json.load(f)
                        temp.append(item)  # 合并字典
                else:
                    temp = [item]

                # 写入新数据
                with open(file_new, "w", encoding="utf-8") as f:
                    json.dump(temp, f, ensure_ascii=False, indent=4)
            else:
                print("已经清理过")


if __name__==main():
    main()