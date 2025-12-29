import random
import os
import json
from tools import Optimize_Q,yolo_to_abs,write_file
from ClassifyTheProblem import classify_question_by_index_withoutimage
from copy import deepcopy
import re

class WithoutImageLabel():
    def __init__(self,Q_list, without_image_template, without_image_path,CategoryPath,
                                    base_url, api_key,model):
        self.QList = Q_list
        self.TemplatesList = without_image_template   # 正常模板
        self.SavePath = without_image_path
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.CategoryPath = CategoryPath

    def GetRandomNum(self,Q_length):
        # list 不包含尾，开区间
        templates_ids = list(range(0,Q_length))
        # 阶段识别、异常检测、特定异常确认、通用异常询问、异常原因分析
        weights = [9,9,9,9,9,9,1,9,9,9,9,9,9,9]
        weights= weights[:Q_length]

        # 生成一个加权随机数
        result = random.choices(templates_ids, weights=weights, k=1)[0]
        return result

    def get_question(self,existed=None, first=None):
        # randint 是闭区间，头尾包含
        templates_id = self.GetRandomNum(len(self.QList))
        # templates_id = random.randint(0, 2)
        if existed:
            while templates_id in existed or self.QList[templates_id]=="temp1":
                templates_id = self.GetRandomNum(len(self.QList))
            existed.append(templates_id)

        old_question = self.QList[templates_id]
        # while old_question =="temp1":
        #     old_question = self.QList[templates_id]
        category = classify_question_by_index_withoutimage(templates_id)

        print(old_question)
        # old_question = templates[3]
        new_question, token = Optimize_Q(old_question,self.base_url,self.api_key,self.model)
        return new_question, templates_id,category


    def chat_message(self,message):
        turn_flag = random.randint(1, 3)
        message_with_category = deepcopy(message)
        # multi_turn_flag = 1
        print(turn_flag)

        existed = [len(self.QList)]
        # turn_times = 2
        for i in list(range(0, turn_flag)):
            # 获取问题并优化

            new_question, template_id ,category= self.get_question(existed)

            # 保存用户输入
            message.append({"role": "user", "content": new_question})
            message_with_category.append({"role": "user", "content": new_question,"category":category})
            # 获取模型回答response



            # 模板分界线
            boundary = self.QList.index("temp1")

            if template_id < boundary:
                funtion = self.TemplatesList[0]
                client = funtion(self.base_url, self.api_key,self.model)
                response = client.handel(new_question)
            else:
                funtion = self.TemplatesList[1]
                client = funtion(self.base_url, self.api_key,self.model)
                response = client.handel(new_question)

            # response = get_assitant_response(image_path, new_question, label, type)
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
            message.append({"role": "assistant", "content": response})
            message_with_category.append({"role": "assistant", "content": response})
            print(new_question)
            print(response)

        return message,message_with_category


    def start(self,nums):

        while nums>0:
            data = {}
            data["data"] = []
            data_category = {}
            data_category["data"] = []

            message = []
            multi_label = {}
            multi_label_category = {}


            system_str = f'你是一个面向3D打印场景的目标检测大师，具备精准分析缺陷的能力。\n 职能:按照要求详细回答用户咨询的问题。'
            message.append({"role": "system", "content": system_str})
            # print(system_str)

            message,message_with_category= self.chat_message(message)

            multi_label["messages"] = message
            data["data"].append(multi_label)
            write_file(self.SavePath, multi_label, data)

            # 增加一个问题分类，写入新的文件
            multi_label_category["messages"] = message_with_category
            data_category["data"].append(multi_label_category)
            write_file(self.CategoryPath, multi_label_category, data_category)

            nums = nums - 1



