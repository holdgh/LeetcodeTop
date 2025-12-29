import random
import os
import json
from copy import deepcopy
import re
import sys
from tools import Optimize_Q,yolo_to_abs,write_file,HorizontalAndVerticalStripeJudgment
from ClassifyTheProblem import classify_question_by_index_error
label_info = {"0": "刮刀横/竖条纹", "1": "曲翘凸起", "2": "污染物", "3": "污染物", "4": "球化",
                      "5": "刮刀横/竖条纹", "6": "刮刀横/竖条纹","7": "曲翘凸起","8": "曲翘凸起", "9": "铺粉不完全"}
label_error = {"0": "刮刀横条纹", "1": "曲翘凸起", "2": "污染物",  "4": "球化","5": "刮刀竖条纹", "9": "铺粉不完全"}

class ImageErrorLabel():
    def __init__(self,Q_list,list_template,path,CategoryPath,base_url, api_key,model):
        self.QList = Q_list
        self.TemplatesList = list_template  # 异常模板
        self.path = path
        self.CategoryPath = CategoryPath
        self.base_url = base_url
        self.api_key = api_key
        self.model = model

    def change_class(self,label):
        # 确保字典中有至少两个类别
        change_info = {}
        label_dict = deepcopy(label)

        if len(label_dict) < 1:
            print("需要至少1个类别才能进行修改")
            return label_dict

        # 获取所有类别
        source_categories = list(label_dict.keys())
        target_categories = list(label_error.values())

        # 随机选择源类别和目标类别（确保不同）
        source_category = random.sample(source_categories, 1)[0]

        target_category = deepcopy(source_category)
        # 确保
        while target_category == source_category:
        # while target_category in source_categories:
            target_category = random.sample(target_categories, 1)[0]

        # 确保源类别中有box
        if not label_dict[source_category]:
            print(f"类别 '{source_category}' 中没有box")
            return label_dict

        # 随机选择一个box
        box_index = random.randint(0, len(label_dict[source_category]) - 1)
        selected_box = label_dict[source_category].pop(box_index)

        # 将box添加到目标类别
        # 将box添加到目标类别
        if target_category in source_categories:
            label_dict[target_category].append(selected_box)
        else:
            label_dict[target_category] = [selected_box]
        change_info["actual_category"] = source_category
        change_info["predict_category"] = target_category
        change_info["selected_box"] = selected_box

        return label_dict, change_info

    def from_boxes_get_system_msg(self,label, img_width, img_height):

        label_msg = ""

        for classes, boxes in label.items():
            # classes = classes.split("_")[1]
            classes = classes.split("(")[0]
            for box in boxes:
                # box1 = yolo_to_abs(box, img_width, img_height)
                label_msg += f"<box>{box[0]},{box[1]},{box[2]},{box[3]}</box>区域表现的异常类型可能是{classes}\n"

        return label_msg

    def GetRandomNum(self,Q_length):
        # list 不包含尾，开区间
        templates_ids = list(range(0,Q_length))
        # 阶段识别、异常检测、特定异常确认、通用异常询问、异常原因分析
        weights = [5,5,9,1,30]
        weights= weights[:Q_length]

        # 生成一个加权随机数
        result = random.choices(templates_ids, weights=weights, k=1)[0]
        return result

    def get_question(self,existed=None, first=None):

        if first:
            templates_id = self.GetRandomNum(len(self.QList) - 1)
        else:
            templates_id = self.GetRandomNum(len(self.QList))
        if existed:
            while templates_id in existed:
                templates_id = self.GetRandomNum( len(self.QList) - 1)
            existed.append(templates_id)
        templates = self.QList[templates_id]
        # templates = self.QList[3]
        # templates_id = 3
        category = classify_question_by_index_error(templates_id)
        seed = random.randint(0, len(templates) - 1)
        old_question = templates[seed]
        print(old_question)
        # old_question = templates[3]
        new_question, token = Optimize_Q(old_question,self.base_url,self.api_key,self.model)
        return new_question, templates_id,category


    def chat_message(self,message, label, image_path, type,change_info):
        message_with_category = deepcopy(message)

        existed = [len(self.QList)]
        turn_times = random.randint(1, 3)
        print(turn_times)
        for i in list(range(0, turn_times)):

            # 获取问题并优化
            if i == 0:
                new_question, template_id,category = self.get_question(existed, True)  # 优化过后的
                new_question = "<image>" + new_question
            else:
                new_question, template_id ,category= self.get_question(existed)
            # 判断问题分类

            message_with_category.append({"role": "user", "content": new_question,"category":category})

            # 保存用户输入
            message.append({"role": "user", "content": new_question})
            # 获取模型回答response
            funtion = self.TemplatesList[template_id]
            client = funtion(self.base_url, self.api_key,self.model)
            response = client.handel(new_question, label, type)


            # response = get_assitant_response(image_path, new_question, label, type)
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

            message.append({"role": "assistant", "content": response})
            message_with_category.append({"role": "assistant", "content": response})
            print(new_question)
            print(response)

        return message,message_with_category

    def chat_error_message(self,message, label, image_path, type):
        message_with_category = deepcopy(message)

        existed = [len(self.QList)]
        turn_times = random.randint(1, 3)
        print(turn_times)
        for i in list(range(0, turn_times)):

            # 获取问题并优化
            if i == 0:
                new_question, template_id,category = self.get_question(existed, True)  # 优化过后的
                new_question = "<image>" + new_question
            else:
                new_question, template_id ,category= self.get_question(existed)
            # 判断问题分类

            message_with_category.append({"role": "user", "content": new_question,"category":category})

            # 保存用户输入
            message.append({"role": "user", "content": new_question})
            # 获取模型回答response
            if label:
                funtion = self.TemplatesList[template_id]
                client = funtion(self.base_url, self.api_key,self.model)
                response = client.handel(new_question, label, type)
            else:
                funtion = self.TemplatesList[template_id]
                client = funtion(self.base_url, self.api_key,self.model)
                response = client.handel(new_question, None, type)

            # response = get_assitant_response(image_path, new_question, label, type)
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

            message.append({"role": "assistant", "content": response})
            message_with_category.append({"role": "assistant", "content": response})
            print(new_question)
            print(response)

        return message,message_with_category


    def start(self,num):

        data = {}
        data["data"] = []
        data_category = {}
        data_category["data"] = []
        walks = ["yoloLabels_生产第一批","yoloLabels_生产第二批","yoloLabels_公共"]
        for walk in walks:
            for root, dirs, files in os.walk(walk):

                for file in files:
                    # 图片路径整理
                    multi_label = {}
                    multi_label_category = {}
                    label_path = os.path.relpath(os.path.join(root, file)).replace("\\", "/")

                    # label = yolo_label.get(file,None)
                    if "Image" in file:
                        image_name = "Image" + file.split("__")[1].split("Image")[1].split(".")[0] + ".jpg"
                        image_path = os.path.relpath(os.path.join("上飞公司\image", image_name)).replace("\\", "/")
                    elif "铺粉" in file or "打印" in file:
                        image_name = file.split("__")[1].replace("txt","jpg")
                        image_path = os.path.relpath(os.path.join("上飞公司\image2", image_name)).replace("\\", "/")
                    elif "Image" not in file and "__" in file:
                        image_name = file.split("__")[1].split(".")[0] + ".png"
                        image_path = os.path.relpath(os.path.join("橡树岭", image_name)).replace("\\", "/")
                    else:
                        image_name = file.split(".")[0].split("-")
                        image_name = image_name[1] + "-" + image_name[2] + ".png"
                        image_path = os.path.relpath(
                            os.path.join("橡树岭", image_name)).replace("\\", "/")

                    # _path = os.path.relpath(os.path.join("yoloLabels_生产", name))
                    print(image_path, file, image_name, label_path)

                    # 检查这个yololabel是否已经生成过多模态标注
                    need_label_flag = True

                    if os.path.exists(self.path):
                        with open(self.path, "r", encoding="utf-8") as f:
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
                            if len(content) > 0:
                                label = {}
                                for line in content:
                                    if len(line) > 0:
                                        temp = line.split(" ")
                                        classes = label_info.get(temp[0])
                                        boxes = label.get(classes, [])
                                        box = [round(float(temp[1]), 2), round(float(temp[2]), 2),
                                               round(float(temp[3]), 2), round(float(temp[4]), 2)]
                                        if classes=="刮刀横/竖条纹":
                                            # 判断横条纹还是竖条纹
                                            classes= HorizontalAndVerticalStripeJudgment(classes,box)
                                        if "橡树岭" in image_path:
                                            jpg_width, jpg_height = 1842, 1842
                                        else:
                                            jpg_width, jpg_height = 3450, 3450
                                        box = yolo_to_abs(box, jpg_width, jpg_height)
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
                    if "spreaded" in image_name or "Image" in image_name or "铺粉" in image_name:
                        type = "铺粉"
                    else:
                        type = "打印"

                    print(type)
                    # 如果这个图片能找到标注信息
                    if label and need_label_flag and num>0:
                        message = []

                        label_change,change_info = self.change_class(label)
                        print(change_info)
                        label_msg = self.from_boxes_get_system_msg(label_change, jpg_width, jpg_height)
                        system_str = f'你是一个面向3D打印场景的目标检测大师，具备精准分析图像缺陷的能力。\n 职能:从给出的重点区域分析，该区域是否包含异常。给定区域对应的异常类型可能会不正确，你需要先判断区域和类别是否正确，然后回答用户的问题。并且这个判断过程禁止输出。\n{label_msg}。 \n *注意*:<box>x_min,y_min,x_max,y_max</box>区域表示需要重点关注的区域。依次是:左上角x坐标，左上角y坐标，右下角x坐标，右下角y坐标。'
                        message.append({"role": "system", "content": system_str})



                        message,message_with_category = self.chat_message(message, label, image_path, type,change_info)

                        multi_label["messages"] = message
                        multi_label["images"] = [image_path]
                        data["data"].append(multi_label)
                        # 如果文件存在，读取旧数据
                        write_file(self.path, multi_label, data)

                        # 增加一个问题分类，写入新的文件
                        multi_label_category["messages"] = message_with_category
                        multi_label_category["images"] = [image_path]
                        data_category["data"].append(multi_label_category)
                        write_file(self.CategoryPath, multi_label_category, data_category)
                        num = num -1
                    # 图片没有标注信息
                    elif num <1:
                        print("生成指定数量，结束了")
                        sys.exit(0)
                    elif need_label_flag:
                        print("没有标注信息不需要标注")

                    # 图片已经标注过
                    else:
                        print("这个图片已经标注过了")


# def main():
#     label = {'刮刀横/竖条纹': [[0, 465, 3450, 500], [0, 845, 3450, 879], [1190, 258, 1224, 707], [1362, 138, 1397, 828], [1466, 672, 1500, 983], [1569, 344, 1604, 690], [1742, 138, 1776, 828], [1845, 552, 1880, 1173], [2121, 258, 2156, 707], [2052, 569, 2087, 1086], [1983, 327, 2018, 707], [2208, 137, 2277, 1311]], '曲翘凸起': [[897, 1069, 1173, 1345], [828, 1380, 1173, 1656], [948, 1776, 1190, 2018], [1535, 1983, 1776, 2087], [1897, 1949, 2104, 2052]]}
#     change_class(label)
#
# if __name__ == "__main__":
#     main()