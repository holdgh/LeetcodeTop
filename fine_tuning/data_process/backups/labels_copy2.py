from openai import OpenAI
import re
import random
import os
import json
from Multi_model_label.templates.templates import StageIdentification,AbnormalResults,SpecificAbnormalJudgment,AnomalyClassification,CauseAnalysis,Ab_StageIdentification,Ab_AbnormalResults,Ab_SpecificAbnormalJudgment,Ab_AnomalyClassification,Ab_CauseAnalysis
import argparse
from Multi_model_label.templates.templates_without_image import template_1,template_2

count = 0


Q_list = [
    ["图片属于铺粉还是打印阶段的图片？","图片中是否铺粉异常","这张照片拍摄的是铺粉环节还是打印环节？","此图是铺粉错误还是打印错误?","这是铺粉步骤的图片还是打印步骤的图片？","图片属于打印异常吗","这个图片属于增材制造的哪个阶段？","此图是铺粉异常还是打印异常?","请分析该图像对应的激光熔融具体阶段","该图片来自增材制造的哪一步骤","有什么异常"],
    ["图片上可能发生了什么异常？","这张照片是否显示了某种异常","是否有什么问题发生在图片中","图片里面是否发生了异常","图片中哪些位置可能会发生异常","框出图中可能发生问题的部分"],
    ["图片中是否发生了污染物的异常？","图片中是否发生了刮痕的异常？","图片中是否发生了球化的异常？","图片中是否发生了铺粉不完全的异常？","图片中是否发生了翘曲凸起的异常？"],
    ["增材制造的打印阶段可能会检测到什么异常","铺粉阶段可能会发现哪些问题"],# 通用类
    ["发生图片中的异常可能原因是什么？","为什么会发生图片中反映的异常","哪些技术问题可能导致图中出现异常？","为什么会发生刮刀横竖条纹的异常？","为什么会发生污染物的异常？","为什么会发生铺粉不完全的异常？","为什么会发生球化的异常？","为什么会发生翘曲凸起的异常？"]
]

Q_list_without_image = [
    ["增材制造的打印阶段可能会检测到什么异常","铺粉阶段可能会发现哪些问题","激光熔融增材制造包含几个阶段"],# 通用类
    ["异常发生的原因","哪些操作会导致异常","为什么会发生刮刀横竖条纹的异常？","为什么会发生污染物的异常？","为什么会发生铺粉不完全的异常？","为什么会发生球化的异常？","为什么会发生翘曲凸起的异常？"]
]
not_normal_path = "test.json"
normal_path = "test.json"
without_image = "test.json"


Abnormal_list_template = [StageIdentification,AbnormalResults,SpecificAbnormalJudgment,AnomalyClassification,CauseAnalysis]
Normal_list_template = [Ab_StageIdentification,Ab_AbnormalResults,Ab_SpecificAbnormalJudgment,Ab_AnomalyClassification,Ab_CauseAnalysis]
without_image_template = [template_1,template_2]
base_url="https://api.fe8.cn/v1"
api_key="sk-j2n1UvlYoG7zBwkF5DZrlQ6QYeW1qhhEtsWwDFW5oUYystzz"




def Optimize_Q(qus):
    client = OpenAI(base_url="https://api.fe8.cn/v1",api_key="sk-j2n1UvlYoG7zBwkF5DZrlQ6QYeW1qhhEtsWwDFW5oUYystzz")
    # client = ZhipuAI(api_key="4d9215f97db5474893055d9c7d9b4699.aqivdQGDomfJm8WK")
    messages = [


        {
            "role": "system",
            "content": ''' 
            # 角色设定：作为激光熔融打印增材制造的行业专家，并且是一个优化专家，专注于改进用户提供的初始问题表述，保持原意基础上提升表达效果。
            # 注意：
                ## “铺粉”和“上色不是同义词替换”
                ## “3D打印”和“3D打印机”不是同义词替换
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
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    event_text = response.choices[0].message.content if response.choices else ""
    if "question" in event_text:
        sub_question = re.findall(r".*<question>(.*)</question>", event_text, re.DOTALL)[0]
    else:
        print(qus)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )

        event_text = response.choices[0].message.content if response.choices else ""
        sub_question = re.findall(r".*<question>(.*)</question>", event_text, re.DOTALL)[0]

    return sub_question, response.usage.total_tokens

def adjust(data):
    # data:list
    # return data_dict:dict
    data_dict = {}
    for item in data:
        data_dict[item["image"]] = item

    return data_dict

def yolo_to_abs(yolo_box, img_width, img_height):
    x_center, y_center, w, h = yolo_box
    x_min = int((x_center - w / 2) * img_width)
    y_min = int((y_center - h / 2) * img_height)
    x_max = int((x_center + w / 2) * img_width)
    y_max = int((y_center + h / 2) * img_height)
    return [x_min, y_min, x_max, y_max]

def from_boxes_get_system_msg(label,img_width, img_height):
    # annotations = label.get("annotations", None)[0].get("result", None)
    # boxes = {}
    # if annotations:
    #     # print(annotations)
    #     for annotation in annotations:
    #         box = annotation.get("value", None)
    #         rectanglelabels = box["rectanglelabels"][0]
    #         boxes[rectanglelabels] = [box["x"], box["y"], box["width"], box["height"]]
    # # print(boxes)
    label_msg = ""
    # for rectanglelabel, box in boxes.items():
    #     label_msg += f"<box>{box[0]},{box[1]},{box[2]},{box[3]}</box>区域表现的异常类型是{rectanglelabel}\n"

    for classes,boxes in label.items():
        # classes = classes.split("_")[1]
        classes = classes.split("(")[0]
        for box in boxes:
            # box1 = yolo_to_abs(box, img_width, img_height)
            label_msg += f"<box>{box[0]},{box[1]},{box[2]},{box[3]}</box>区域表现的异常类型是{classes}\n"

    return label_msg

def get_question(existed=None,first=None):
    if first:
        templates_id = random.randint(0, len(Q_list) - 2)
    else:
        templates_id = random.randint(0, len(Q_list) - 1)
    if existed:
        while templates_id in existed:
            templates_id = random.randint(0, len(Q_list) - 1)
        existed.append(templates_id)
    templates = Q_list[templates_id]
    # templates = Q_list[3]
    # templates_id = 3
    seed = random.randint(0, len(templates) - 1)
    old_question = templates[seed]
    print(old_question)
    # old_question = templates[3]
    new_question, token = Optimize_Q(old_question)
    return new_question,templates_id

# def get_assitant_response(rel_path,question,label,type):
#     # with open(rel_path, 'rb') as img_file:
#     #     img_base = base64.b64encode(img_file.read()).decode('utf-8')
#     #
#     #     try:
#     #         img_bytes = base64.b64decode(img_base)
#     #         img = Image.open(io.BytesIO(img_bytes))
#     #         img.verify()  # 验证图片是否损坏
#     #
#     #     except Exception as e:
#     #         print(f"图片验证失败: {e}")
#
#         response, token = get_response(question, label, type)
#         return(response)


# 写如文件
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

#
def chat_message(message,label,image_path,type):
    multi_turn_flag = random.randint(0,1)
    # multi_turn_flag = 1
    print(multi_turn_flag)
    # 多轮对话
    if multi_turn_flag:
        existed = [len(Q_list)]
        turn_times = random.randint(2,3)
        # turn_times = 2
        for i in list(range(0,turn_times)):
            # 获取问题并优化
            if i == 0:
                new_question,template_id = get_question(existed,True)  # 优化过后的
                new_question = "<image>"+ new_question
            else:
                new_question,template_id = get_question(existed)

            # 保存用户输入
            message.append({"role": "user", "content": new_question})
            # 获取模型回答response
            if label:
                funtion = Abnormal_list_template[template_id]
                client = funtion(base_url, api_key)
                response = client.handel(new_question, label, type)
            else:
                funtion = Normal_list_template[template_id]
                client = funtion(base_url, api_key)
                response = client.handel(new_question,None,type)

            # response = get_assitant_response(image_path, new_question, label, type)
            message.append({"role": "assistant", "content": response})
            print(new_question)
            print(response)
    else:

        new_question, template_id = get_question()  # 优化过后的
        new_question = "<image>" + new_question
        # 保存用户输入
        message.append({"role": "user", "content": new_question})
        if label:
            funtion = Abnormal_list_template[template_id]
            client = funtion(base_url, api_key)
            response = client.handel(new_question, label, type)
        else:
            funtion = Normal_list_template[template_id]
            client = funtion(base_url, api_key)
            response = client.handel(new_question,None,type)
        # response = get_assitant_response(image_path, new_question, label, type)
        message.append({"role": "assistant", "content": response})
        print(new_question)
        print(response)

    return message

def message_without_image():
    message = []
    system_str = f'你是一个面向3D打印场景的目标检测大师，具备精准识别、定位图像中固定位置图像是否包含缺陷的能力。\n 职能:能为用户提供专业的异常分析。'
    message.append({"role": "system", "content": system_str})
    # print(system_str)



    multi_label["messages"] = message
    multi_label["images"] = [image_path]
    data["data"].append(multi_label)

    # 如果文件存在，读取旧数据

    # print(data)
    write_file(normal_path, multi_label, data)


def message_with_image():
    label_info = {"0": "刮刀横/竖条纹", "1": "曲翘凸起", "2": "污染物", "3": "污染物", "4": "球化",
                  "5": "刮刀横/竖条纹", "6": "曲翘凸起", "7": "铺粉不完全"}
    data = {}
    data["data"] = []
    walks = ["yoloLabels_生产", "yoloLabels_公共"]
    for walk in walks:
        print(walk)
        for root, dirs, files in os.walk(walk):
            for file in files:
                # 图片路径整理
                multi_label = {}
                label_path = os.path.relpath(os.path.join(root, file)).replace("\\","/")

                # label = yolo_label.get(file,None)
                if "Image" in file:
                    image_name = "Image"+file.split("__")[1].split("Image")[1].split(".")[0]+".jpg"
                    image_path = os.path.relpath(os.path.join("上飞公司\image", image_name)).replace("\\","/")
                elif "Image" not in file and "__" in file:
                    image_name = file.split("__")[1].split(".")[0] + ".png"
                    image_path = os.path.relpath(os.path.join("橡树岭", image_name)).replace("\\", "/")
                else:
                    image_name = file.split(".")[0].split("-")
                    image_name = image_name[1]+"-"+ image_name[2]+ ".png"
                    image_path = os.path.relpath(
                        os.path.join("橡树岭", image_name)).replace("\\", "/")

                # _path = os.path.relpath(os.path.join("yoloLabels_生产", name))
                print( image_path, file,image_name,label_path)

                # 检查这个yololabel是否已经生成过多模态标注
                need_label_flag = True

                if os.path.exists(not_normal_path):
                    with open(not_normal_path, "r", encoding="utf-8") as f:
                        old_data = json.load(f)
                        for item in old_data:
                            if image_path in item["images"]:
                                need_label_flag = False
                                break

                if os.path.exists(normal_path):
                    with open(normal_path, "r", encoding="utf-8") as f:
                        old_data = json.load(f)
                        for item in old_data:
                            if image_path in item["images"]:
                                need_label_flag = False
                                break




                # 根据图片名称获取标注
                if os.path.exists(label_path):
                    with open(label_path, "r", encoding="utf-8") as file:
                        content = file.read()  # 读取全部文本
                        content = content.split("\n")
                        if len(content)>0:

                            label = {}
                            for line in content:
                                if len(line)>0:
                                    temp = line.split(" ")
                                    classes = label_info.get(temp[0])
                                    # 注意：这里是因为在实验数据集中有人使用了生产数据集的标签，所以这里增加了判断，如果用的生产标签，就替换为实验的
                                    # if classes == "生产_刮刀横/竖条纹":
                                    #     classes = "公共_刮刀横/竖条纹"
                                    boxes = label.get(classes,[])
                                    box = [round(float(temp[1]), 2),round(float(temp[2]), 2),round(float(temp[3]), 2),round(float(temp[4]), 2)]
                                    if "橡树岭" in image_path:
                                        jpg_width, jpg_height = 1842, 1842
                                    else:
                                        jpg_width, jpg_height = 3450, 3450
                                    box = yolo_to_abs(box,jpg_width, jpg_height)
                                    boxes.append(box)
                                    label[classes] = boxes
                        else:
                            label = None
                            print("标签为空")
                    print(label)
                else:
                    label = None
                    print("路径不存在")



                #  根据图片名称判断图片属于哪一个阶段的图片。
                if "spreaded" in image_name or "Image" in image_name:
                    type = "铺粉"
                else:
                    type = "打印"
                # 如果这个图片能找到标注信息
                if label and need_label_flag:
                    message = []

                    label_msg = from_boxes_get_system_msg(label,jpg_width, jpg_height)
                    system_str = f'你是一个面向3D打印场景的目标检测大师，具备精准识别、定位图像中固定位置图像是否包含缺陷的能力。\n 职能:从给出的重点区域分析，该区域是否包含异常。\n{label_msg}。 \n *注意*:<box>x,y,w,h</box>区域表示需要重点关注的区域。依次是:左上角x坐标，左上角y坐标，右下角x坐标，右下角y坐标。'
                    message.append({"role": "system", "content": system_str})
                    # print(system_str)

                    message = chat_message(message,label,image_path,type)

                    multi_label["messages"] = message
                    multi_label["images"] = [image_path]
                    data["data"].append(multi_label)

                        # 如果文件存在，读取旧数据
                    write_file(not_normal_path, multi_label, data)
                # 图片没有标注信息
                elif need_label_flag:
                    print("这个图片没有标注信息但是需要标注，这是一个正常样本")
                    message = []
                    system_str = f'你是一个面向3D打印场景的目标检测大师，具备精准识别、定位图像中固定位置图像是否包含缺陷的能力。\n 职能:从给出的重点区域分析，该区域是否包含异常。\n在这个图片中没有发生异常。请按照没有异常发生的情况回答用户的问题。'
                    message.append({"role": "system", "content": system_str})
                    # print(system_str)
                    message = chat_message(message, None,image_path,type)

                    multi_label["messages"] = message
                    multi_label["images"] = [image_path]
                    data["data"].append(multi_label)

                        # 如果文件存在，读取旧数据

                    # print(data)
                    write_file(normal_path,multi_label, data)



                # 图片已经标注过
                else:
                    print("这个图片已经标注过了")


def main():
    count = 0
    # 注意：如果要用这个脚本跑生产数据集的标签，需要先对齐标签顺序和内容
    #
    #

    parser = argparse.ArgumentParser(description="示例脚本")
    parser.add_argument("--with_image", action="store_true", help="是否输入图片")

    args = parser.parse_args()
    if args.with_image:
        message_with_image()
    else:
        message_without_image()







if __name__ ==main():
    main()