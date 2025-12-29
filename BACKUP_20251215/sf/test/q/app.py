from flask import Flask, request, jsonify, send_file, send_from_directory
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from openai import OpenAI
import tempfile
import os

app = Flask(__name__, static_folder='static')  # 指定静态文件目录

# 其他代码保持不变...

# 添加根路由，返回静态 HTML 文件
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


# 初始化模型
yolo_model = YOLO('/home/liuyl/sf/test/best.pt')
client = OpenAI(
    api_key="0",
    base_url="http://0.0.0.0:6661/v1"
)

# 类别名称
class_names = ["刮刀横/竖条纹", "曲翘凸起", "污染物", "污染物", "球化", "曲翘凸起", "铺粉不完全"]

# 辅助函数：将YOLO格式转换为像素坐标
def yolo_xywh2mllm_xy(input_image, box):
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
    return [class_id, x1, y1, x2, y2, confidence]

# 辅助函数：绘制检测框
def draw_yolo_boxes(input_image, yolo_boxes, class_names):
    image = cv2.imread(input_image)
    height, width, _ = image.shape
    # 为每个类别生成随机颜色
    np.random.seed(42)  # 设置随机种子，确保颜色一致
    colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype=np.uint8)
    # 将OpenCV图像转换为PIL图像
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/SimHei.ttf", 100)  

    # 绘制每个边界框
    for box in yolo_boxes:
        box = yolo_xywh2mllm_xy(input_image, box)
        class_id = int(box[0])
        confidence = box[5]
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
        
        # 绘制边界框
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=5)
        
        # 准备标签文本
        label = f"{class_names[class_id]}"
        if confidence < 1.0:
            label += f" {confidence:.2f}"

        # 标签背景和文本
        bbox = font.getbbox(label)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.rectangle([(x1, y1 - text_height - 5), (x1 + text_width, y1)], fill=color)
        draw.text((x1, y1 - text_height - 2), label, font=font, fill=(255, 255, 255))
    
    # 将PIL图像转换回OpenCV格式
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return image

# 辅助函数：格式化YOLO检测结果为提示词
def format_yolo_detections(input_image, detections, class_names):
    base_template = """你是一个面向3D打印场景的目标检测大师，具备精准识别、定位图像中固定位置图像是否包含缺陷的能力。\n 职能:从给出的重点区域分析，该区域是否包含异常。"""
    
    for det in detections:
        box = yolo_xywh2mllm_xy(input_image, det)
        class_id = box[0]
        x1 = box[1]
        y1 = box[2]
        x2 = box[3]
        y2 = box[4]
        class_name = class_names[int(class_id)]
        # 添加到模板
        base_template += f"\n<box>{x1},{y1},{x2},{y2}</box>区域表现的异常类型是{class_name}"
    
    base_template += "\n\n*注意*:<box>x1,y1,x2,y2</box>区域表示需要重点关注的区域。依次是:左上角x坐标，左上角y坐标，右下角x坐标，右下角y坐标\n\n"
    return base_template

# API端点：执行质检分析
@app.route('/api/inspect', methods=['POST'])
def inspect_image():
    try:
        # 获取上传的图像
        if 'image' not in request.files:
            return jsonify({"error": "未上传图像"}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"error": "未选择图像文件"}), 400
        
        # 保存临时图像文件
        temp_image = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        image_file.save(temp_image.name)
        temp_image.close()
        
        # 获取用户问题
        question = request.form.get('question', '')
        if not question:
            return jsonify({"error": "请提供质检问题"}), 400
        
        # 步骤1: 调用YOLO模型检测异常
        results = yolo_model.predict(
            source=temp_image.name,
            project='/home/liuyl/sf/results',
            name='my_predict',
            conf=0.4,
            save=False,
            save_txt=False
        )
        
        detections = []
        for result in results:
            for box in result.boxes:
                cls_idx = int(box.cls)
                # 获取边界框坐标（归一化xywh格式,保留6位小数）
                x_center, y_center, w, h = [round(float(val), 6) for val in box.xywhn[0].tolist()]
                confidence = round(float(box.conf), 6)
                detections.append([cls_idx, x_center, y_center, w, h, confidence])
        
        # 步骤2: 生成多模态大模型的提示词
        prompt = format_yolo_detections(temp_image.name, detections, class_names)
        
        # 步骤3: 绘制检测结果图像
        result_image = draw_yolo_boxes(temp_image.name, detections, class_names)
        
        # 保存结果图像到临时文件
        temp_result_image = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        cv2.imwrite(temp_result_image.name, result_image)
        temp_result_image.close()
        
        # 步骤4: 调用多模态大模型进行分析
        mllm_model_path = '/home/liuyl/llama_factory/LLaMA-Factory/output/qwen2.5_vl_lora_sft'
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": prompt}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"file://{temp_image.name}"
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
        
        analysis_response = result.choices[0].message.content
        
        # 准备返回结果
        formatted_detections = []
        for det in detections:
            class_id = int(det[0])
            formatted_detections.append({
                "class_id": class_id,
                "class_name": class_names[class_id],
                "confidence": det[5]
            })
        
        # 构建响应
        response = {
            "detections": formatted_detections,
            "analysis": analysis_response,
            "result_image_url": f"/api/result_image/{os.path.basename(temp_result_image.name)}"
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # 清理临时文件
        if 'temp_image' in locals():
            os.unlink(temp_image.name)
        if 'temp_result_image' in locals():
            os.unlink(temp_result_image.name)

# API端点：获取结果图像
@app.route('/api/result_image/<filename>')
def get_result_image(filename):
    result_path = os.path.join(tempfile.gettempdir(), filename)
    if os.path.exists(result_path):
        return send_file(result_path, mimetype='image/jpeg')
    else:
        return jsonify({"error": "图像不存在"}), 404


# 其他 API 路由保持不变...

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)