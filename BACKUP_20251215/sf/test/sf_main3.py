import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
# from predict import image_predict
from ultralytics import YOLO
from openai import OpenAI

def image_predict(yolo_model,input_image):
    results = yolo_model.predict(
        source = input_image,
        project = '/home/liuyl/sf/results',
        name = 'my_predict',  # 绝对路径，包含完整层级
        conf = 0.4,
        save = True,
        save_txt = True
        # save_conf = True
    )
    detections  = []
    for result in results:
        for box in result.boxes:
            cls_idx = int(box.cls)
            # 获取边界框坐标（归一化xywh格式,保留6位小数）
            x_center, y_center, w, h = [round(float(val), 6) for val in box.xywhn[0].tolist()]
            detections.append([cls_idx,x_center, y_center, w, h])
    return detections

def yolo_xywh2mllm_xy(input_image,box):
    image = cv2.imread(input_image)
    height, width, _ = image.shape
    class_id = int(box[0])
    x_center, y_center, w, h = box[1:5]
    confidence = box[5] if len(box) == 6 else 1.0
    # 将YOLO格式转换为像素坐标
    x1 = int((x_center - w/2) * width)
    y1 = int((y_center - h/2) * height)
    x2 = int((x_center + w/2) * width)
    y2 = int((y_center + h/2) * height)
    return [class_id,x1,y1,x2,y2]


def draw_yolo_boxes(input_image, yolo_boxes, output_path=None, class_names=None, colors=None):
    image = cv2.imread(input_image)
    height, width, _ = image.shape
    # 为每个类别生成随机颜色
    if colors is None:
        np.random.seed(42)  # 设置随机种子，确保颜色一致
        colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype=np.uint8)
    # 将OpenCV图像转换为PIL图像
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/SimHei.ttf", 100)  
    # 绘制每个边界框
    for box in yolo_boxes:
        box = yolo_xywh2mllm_xy(input_image,box)
        class_id = int(box[0])
        confidence = box[5] if len(box) == 6 else 1.0
        x1 = box[1]
        y1 = box[2]
        x2 = box[3]
        y2 = box[4]

        # 确保坐标在图像范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width - 1, x2)
        y2 = min(height - 1, y2)
        # 获取类别颜色
        color = tuple(int(c) for c in colors[class_id % len(colors)])
        # 绘制边界框（使用PIL）
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=5)
        # 准备标签文本
        label = f"{class_names[class_id]}"
        if confidence < 1.0:
            label += f" {confidence:.2f}"

        # 尝试使用getbbox()方法
        bbox = font.getbbox(label)  # 长度是4
        # 正确的bbox格式: (left, top, right, bottom)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        # 标签背景
        draw.rectangle([(x1, y1 - text_height - 5), (x1 + text_width, y1)], fill=color)
        # 标签文本
        draw.text((x1, y1 - text_height - 2), label, font=font, fill=(255, 255, 255))
    
    # 将PIL图像转换回OpenCV格式
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, image) # save
    print("检测结果保存在：", output_path)
    # cv2.imshow("sf", image)
    return image


def classify_question_intent(question: str) -> str:
    # 意图识别提示词模板
    INTENT_PROMPT_TEMPLATE = """
    你是一个工业智能问答系统中的意图识别模块，请你根据用户输入的问题，判断其意图属于以下三类之一：

    1. defect：关于图片中是否有异常、问题、缺陷、图片处于哪个阶段、缺陷产生的原因等质检相关的问题；
    2. model：关于你是谁、你是什么模型、你是哪个公司的等模型身份类问题；
    3. other：和质检无关的内容，例如生活、娱乐等内容。
    请你直接输出标签（defect/model/other），不要输出其他内容。

    问题：{question}
    答案：
    """
    client = OpenAI(
        api_key="sk-j2n1UvlYoG7zBwkF5DZrlQ6QYeW1qhhEtsWwDFW5oUYystzz",
        # base_url="https://api.fe8.cn/v1"
        base_url='http://192.168.0.194:6662/v1'
        
    )
    
    prompt = INTENT_PROMPT_TEMPLATE.format(question=question)
    response = client.chat.completions.create(
        # model="gpt-4",
        model='Qwen3-32B',
        messages=[
            {"role": "user", "content": prompt}
        ],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}, # 禁止qwen输出<think>部分
        temperature=0.7
        
    )
    result = response.choices[0].message.content.strip().lower()
    return result

def chat(prompt, input_image, question, mllm_model_path):
    client = OpenAI(
        api_key="0",
        base_url="http://0.0.0.0:6661/v1"
    )
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": prompt}]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",  # 注意 必须是 "image_url"
                    "image_url": {
                        "url": input_image
                    }
                },
                {"type": "text", "text": question}
            ]
        }
    ]

    result = client.chat.completions.create(
        model=mllm_model_path,
        messages=messages,
        max_tokens=1024
    )
    return result.choices[0].message.content


def format_yolo_detections(input_image,detections, class_names):
    """将YOLO检测结果格式化为特定字符串模板"""
    base_template = """你是一个面向3D打印场景的目标检测大师，具备精准识别、定位图像中固定位置图像是否包含缺陷的能力。\n 职能:从给出的重点区域分析，该区域是否包含异常。"""
    
    for det in detections:
        box = yolo_xywh2mllm_xy(input_image,det)
        class_id=box[0]
        x1 = box[1]
        y1 = box[2]
        x2 = box[3]
        y2 = box[4]
        class_name = class_names[int(class_id)]
        # 添加到模板
        base_template += f"\n<box>{x1},{y1},{x2},{y2}</box>区域表现的异常类型是{class_name}"
    
    base_template += "\n\n*注意*:<box>x1,y1,x2,y2</box>区域表示需要重点关注的区域。依次是:左上角x坐标，左上角y坐标，右下角x坐标，右下角y坐标\n\n"
    return base_template

def analyze(question,image):
    if image is None:
        intent = classify_question_intent(question)
        if intent == "model":
            response = "我是 上海联通 和 上飞公司航研所 共同打造的的“天眼”多模态质检大模型"
            
        elif intent == "defect":
            response = "请上传图片"
        else:
            response = "抱歉，我是为增材制造质检量身打造的多模态AI助手，您问的问题我暂时无法回答。" 
        return {
        "intent": intent,
        "answer": response,
        "original_image_url": None,
        "yolo_output_image_url": None,
        "yolo_empty": True
            }
    else:
        class_names = ["刮刀横/竖条纹", "曲翘凸起", "污染物", "污染物", "球化", "曲翘凸起", "铺粉不完全"]
        yolo_model = YOLO('/home/liuyl/sf/test/best.pt')
        # image_url = '/root/home/上飞/实验数据/橡树岭/450-spreaded.png'
        image_url = image
        output_path = "/home/liuyl/sf/image_rectangel/output.jpg"
        mllm_model_path = '/nas_data/lyl_project/llamafactory/LLaMA-Factory/output/qwen2.5_vl_32b_4data_lora_sft'
        yolo_boxes = image_predict(yolo_model, image_url)
        draw_yolo_boxes(image_url, yolo_boxes, output_path, class_names)
        yolo_empty = len(yolo_boxes) == 0
        print('yolo_empty',yolo_empty)
        
        intent = classify_question_intent(question)
        print('intent',intent)
        if intent == "model":
            response = "我是 上海联通 和 上飞公司航研所 共同打造的的“天眼”多模态质检大模型"
            return {
                "intent": intent,
                "answer": response,
                # "original_image_url": f"/static/uploads/{filename}",
                "yolo_output_image_url": None,
                "yolo_empty": yolo_empty
                    }
        elif intent == "defect" and len(yolo_boxes) == 0:
            prompt = "你是一个面向3D打印场景的目标检测大师，具备精准识别、定位图像中固定位置图像是否包含缺陷的能力。\n 职能:从给出的重点区域分析，该区域是否包含异常。"
            response = chat(prompt, image_url, question, mllm_model_path)
            # response = chat2(prompt, question, mllm_model_path)

            return {
                "intent": intent,
                "answer": response,
                # "original_image_url": f"/static/uploads/{filename}",
                "yolo_output_image_url": None,
                "yolo_empty": yolo_empty
                    }
        elif intent == "defect" and len(yolo_boxes) != 0:
            prompt = format_yolo_detections(image_url, yolo_boxes, class_names)
            response = chat(prompt, image_url, question, mllm_model_path)
            return {
                "intent": intent,
                "answer": response,
                # "original_image_url": f"/static/uploads/{filename}",
                "yolo_output_image_url": f"/static/output.jpg",
                "yolo_empty": yolo_empty
                    }
        else:
            response = "抱歉，我是为增材制造质检量身打造的多模态AI助手，您问的问题我暂时无法回答。"
            return {
                "intent": intent,
                "answer": response,
                # "original_image_url": f"/static/uploads/{filename}",
                "yolo_output_image_url": None,
                "yolo_empty": yolo_empty
                    }


question = '图片有什么异常'
image = '/root/home/上飞/实验数据/橡树岭/450-spreaded.png'
aa=analyze(question,image)
print(aa)
