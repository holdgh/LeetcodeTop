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



mllm_model_path = '/home/liuyl/llama_factory/LLaMA-Factory/output/qwen2.5_vl_lora_sft_v2'
# input_image = '/root/home/上飞/实验数据/上飞公司/image/Image_20241028172015880_c.jpg'

input_image = '/root/home/上飞/实验数据/橡树岭/450-spreaded.png'
# input_image ='/home/liuyl/sf/test/1748237865281.jpg'
output_image = "/home/liuyl/sf/image_rectangel/output.jpg"
question = '这个图片是增材制造中的哪个阶段？'


'''
step1: 先调用yolo模型检测异常类型以及位置
'''
yolo_model = YOLO('/home/liuyl/sf/test/best.pt')
yolo_boxes=image_predict(yolo_model,input_image)


'''
step2: 根据yolo检测出的异常类型和box生成mllm的system提示词
'''
# class_names = ["公共_刮刀横/竖条纹", "公共_曲翘凸起", "公共_污染物(打印)", "公共_污染物(铺粉)", "公共_球化", "生产_曲翘凸起", "生产_铺粉不完全"]
class_names = ["刮刀横/竖条纹", "曲翘凸起", "污染物", "污染物", "球化", "曲翘凸起", "铺粉不完全"]
prompt = format_yolo_detections(input_image,yolo_boxes, class_names)
print("\n\nsystem:",prompt)


'''
step3: 保存yolo识别结果图片（已更改标签）
'''
draw_yolo_boxes(input_image=input_image, yolo_boxes=yolo_boxes, output_path=output_image, class_names=class_names)

'''
step4: 将图片、问题、及system提示词送给微调后的多模态大模型进行回答
'''
print('yolo_boxes',yolo_boxes)
if len(yolo_boxes) == 0:
    response='该图片无异常'
else:
    response=chat(prompt, input_image, question, mllm_model_path)
print("question:",question)
print("response:", response)

