import cv2
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from openai import OpenAI

def image_predict(yolo_model,input_image):
    results = yolo_model.predict(
        source = input_image,
        conf = 0.4,
        save = True,
        save_txt = True,
        save_conf = True
    )
    detections  = []
    for result in results:
        for box in result.boxes:
            # 类别id
            cls_idx = int(box.cls)
            # 置信度
            # confidence = float(box.conf)
            confidence = f"{box.conf.item():.2f}"
            print('confidence',confidence)
            # 获取边界框坐标（归一化xywh格式,保留6位小数）
            x_center, y_center, w, h = [round(float(val), 6) for val in box.xywhn[0].tolist()]
            detections.append([cls_idx,x_center, y_center, w, h, confidence])
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
    return [class_id,x1,y1,x2,y2,confidence]


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
        color = tuple(int(c) for c in colors[class_id % len(colors)])
        # 绘制边界框
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=5)
        # 准备标签文本
        label = f"{class_names[class_id]}"

        # 保存图片的类别标签是否加上置信度
        confidence = float(confidence)
        if confidence < 1.0:
            # label += f" {confidence:.2f}"
            label += confidence
        
 

        # 尝试使用getbbox()方法
        bbox = font.getbbox(label) 
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


yolo_model = YOLO('/home/liuyl/sf/test/best_2.pt')
class_names = ["刮刀横/竖条纹", "曲翘凸起", "污染物", "污染物", "球化", "刮刀横/竖条纹", "刮刀横/竖条纹", "曲翘凸起", "曲翘凸起","铺粉不完全"]



# 批量检测整个文件夹中的图片

# files = os.listdir('/home/liuyl/sf/image_yolo_test')
# for file in files:
#     file_path = os.path.join('/home/liuyl/sf/image_yolo_test', file)
#     if os.path.isfile(file_path):
#         input_image = file_path
#         output_image = f"/home/liuyl/sf/image_rectangel/output_{file}"
#         yolo_boxes=image_predict(yolo_model,input_image)
#         draw_yolo_boxes(input_image=input_image, yolo_boxes=yolo_boxes, output_path=output_image, class_names=class_names)


# 检测单个图片

input_image = '/root/home/上飞/实验数据/橡树岭/450-spreaded.png'
output_image = "/home/liuyl/sf/image_rectangel/output.jpg"
yolo_boxes=image_predict(yolo_model,input_image)
draw_yolo_boxes(input_image=input_image, yolo_boxes=yolo_boxes, output_path=output_image, class_names=class_names)