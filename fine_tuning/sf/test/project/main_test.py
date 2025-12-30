# 未解决：不上传图片仅问问题
# 已解决：意图识别模块;异常图片调添加nomal微调模型输出
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from PIL import Image
import uuid
import os
from openai import OpenAI
import os
from sf_question import (classify_question_intent,question_analyze)
from yolo_script import (
    image_predict, draw_yolo_boxes, format_yolo_detections, chat,chat2,
    YOLO, class_names, mllm_model_path
)

app = FastAPI()
os.makedirs("static/uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

yolo_model = YOLO('/home/liuyl/sf/test/best.pt')

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index_test.html", {"request": request})

@app.post("/analyze/")
async def analyze(image: UploadFile = File(...), question: str = Form(...)):
    filename = f"{uuid.uuid4()}.jpg"
    filepath = f"static/uploads/{filename}"
    with open(filepath, "wb") as f:
        f.write(await image.read())
    image_url = os.path.abspath(filepath)
    output_path = "static/output.jpg"

    yolo_boxes = image_predict(yolo_model, image_url)
    
    draw_yolo_boxes(image_url, yolo_boxes, output_path, class_names)
    yolo_empty = len(yolo_boxes) == 0
    print('yolo_empty',yolo_empty)
    
    intent = classify_question_intent(question)
    print('intent',intent)
    if intent == "model":
        response = "我是 上海联通 和 上飞公司航研所 共同打造的的“灵镜”多模态质检大模型"
        return {
            "intent": intent,
            "answer": response,
            "original_image_url": f"/static/uploads/{filename}",
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
            "original_image_url": f"/static/uploads/{filename}",
            "yolo_output_image_url": None,
            "yolo_empty": yolo_empty
                }
    elif intent == "defect" and len(yolo_boxes) != 0:
        prompt = format_yolo_detections(image_url, yolo_boxes, class_names)
        response = chat(prompt, image_url, question, mllm_model_path)
        return {
            "intent": intent,
            "answer": response,
            "original_image_url": f"/static/uploads/{filename}",
            "yolo_output_image_url": f"/static/output.jpg",
            "yolo_empty": yolo_empty
                }
    else:
        response = "抱歉，我是为增材制造质检量身打造的多模态AI助手，您问的问题我暂时无法回答。"
        return {
            "intent": intent,
            "answer": response,
            "original_image_url": f"/static/uploads/{filename}",
            "yolo_output_image_url": None,
            "yolo_empty": yolo_empty
                }


