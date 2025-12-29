from openai import OpenAI
import re
import os
import json
import random
temperature = 0.7

def yolo_to_abs(yolo_box, img_width, img_height):
    x_center, y_center, w, h = yolo_box
    x_min = int((x_center - w / 2) * img_width)
    y_min = int((y_center - h / 2) * img_height)
    x_max = int((x_center + w / 2) * img_width)
    y_max = int((y_center + h / 2) * img_height)
    return [x_min, y_min, x_max, y_max]


def Optimize_Q(qus,base_url,api_key,model):
    client = OpenAI(base_url=base_url, api_key=api_key)
    # client = ZhipuAI(api_key="4d9215f97db5474893055d9c7d9b4699.aqivdQGDomfJm8WK")
    messages = [

        {
            "role": "system",
            "content": ''' 
            # 角色设定：作为激光熔融打印增材制造的行业专家，并且是一个优化专家，专注于改进用户提供的初始问题表述，保持原意基础上提升表达效果。
            # 注意：
                ## “铺粉”和“上色不是同义词替换”
                ## 禁止使用“工艺”加入问题
                ## “3D打印”和“3D打印机”不是同义词替换
                ## 禁止替换原始问题中的异常种类：例如："刮刀横/竖条纹",  "曲翘凸起", "污染物",  "球化","铺粉不完全"
                ## 原始问题中询问可能发生的异常时，禁止给出举例的异常。
                ## 保持原始问题的语义，例如问题属于验证类或询问类，保持问题的语义不变
                ## 增材制造过程分为两个阶段：铺粉阶段和打印阶段，不能简单把打印阶段理解为3D打印或增材制造。

            # 优化准则：

                ## 核心原则 - 通过调整措辞、重组句式等策略重构问题，严格保持原有信息边界，不要脱离行业领域。
                ## 禁止事项 - 避免引入任何虚构或假设性信息。避免导致问题意思发生混乱。
                ## 优化后的新问题放在标签<question></question>中返回。
                ## 必须返回一个问题
                ## 优化的问题字数在100字以内。

                '''
        },
        {
            "role": "user",
            "content": qus

        }

    ]
    try :
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )

        event_text = response.choices[0].message.content if response.choices else ""
        sub_question = re.findall(r".*<question>(.*)</question>", event_text, re.DOTALL)[0]
        token = response.usage.total_tokens
    except :
        print("大模型优化问题发生了问题，没有返回问题")
        sub_question = qus
        token = 0

    return sub_question, token

def write_file(file_path,multi_label,data):
    print("write")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            old_data = json.load(f)
            old_data.append(multi_label)  # 合并字典
    else:
        old_data = data["data"]

        # 写入新数据
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(old_data, f, ensure_ascii=False, indent=4)

# 生成一个随机的大模型回答模板id
def random_temp():
    templates_ids = list(range(1, 5))
    # 阶段识别、异常检测、特定异常确认、通用异常询问、异常原因分析
    weights = [4, 4, 9, 9]

    # 生成一个加权随机数
    result = random.choices(templates_ids, weights=weights, k=1)[0]
    return result


def HorizontalAndVerticalStripeJudgment(cls,box):
    return "刮刀竖条纹" if box[2] < 0.3 else "刮刀横条纹"