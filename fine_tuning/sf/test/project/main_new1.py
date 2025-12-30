
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
from typing import Optional
from sf_question import (classify_question_intent, qwen3_AM_knowledge)
from yolo_script import (
    image_predict, draw_yolo_boxes, format_yolo_detections, InterVL_image_chat, InterVL_chat, Qwen_image_chat, Qwen_chat,
    YOLO
)

app = FastAPI()
os.makedirs("static/uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
yolo_model = YOLO('/home/liuyl/sf/test/best_1.pt')
class_names = ["刮刀横/竖条纹", "曲翘凸起", "污染物", "污染物", "球化", "刮刀横/竖条纹","曲翘凸起", "铺粉不完全"]
# class_names = ["刮刀横/竖条纹", "曲翘凸起", "曲翘凸起", "污染物", "球化","刮刀横/竖条纹", "球化", "铺粉不完全"]
# mllm_model_path = '/home/liuyl/llama_factory/LLaMA-Factory/output/qwen2.5_vl_lora_sft'
mllm_model_path = '/nas_data/lyl_project/llamafactory/LLaMA-Factory/output/qwen2.5_vl_7b_4data_lora_sft'
# mllm_model_path ='InterVL38B'
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index_new1.html", {"request": request})

@app.post("/analyze/")
async def analyze(question: str = Form(...),
                  image: Optional[UploadFile] = File(None)
                  ):
    if image is None:
        intent = classify_question_intent(question)
        if intent == "model":
            response = "我是“上海联通”和“上飞公司航研所”共同打造的的“灵镜”多模态质检大模型"
        elif intent == "image_defect":
            response = "请上传图片"
        elif intent == "defect":
            # response = "请上传图片"
            prompt = "你是一个面向3D打印场景的目标检测大师，具备精准识别、定位图像中固定位置图像是否包含缺陷的能力。\n 职能:按照要求详细回答用户咨询的问题。"
            response = Qwen_chat(prompt, question, mllm_model_path)
            # response = qwen3_AM_knowledge(question)

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
            response = "我是“上海联通”和“上飞公司航研所”共同打造的的“灵镜”多模态质检大模型"
            return {
                "intent": intent,
                "answer": response,
                "original_image_url": f"/static/uploads/{filename}",
                "yolo_output_image_url": None,
                "yolo_empty": yolo_empty
                    }
        elif intent == "defect":
            prompt = "你是一个面向3D打印场景的目标检测大师，具备精准识别、定位图像中固定位置图像是否包含缺陷的能力。\n 职能:从给出的重点区域分析，该区域是否包含异常。"
            # # response = chat(prompt, image_url, question, mllm_model_path)
            response = Qwen_chat(prompt, question, mllm_model_path)
            # response = qwen3_AM_knowledge(question)

            return {
                "intent": intent,
                "answer": response,
                "original_image_url": f"/static/uploads/{filename}",
                "yolo_output_image_url": None,
                "yolo_empty": yolo_empty
                    }
        elif intent == "image_defect" and len(yolo_boxes) != 0:
            prompt = format_yolo_detections(image_url, yolo_boxes, class_names)
            response = Qwen_image_chat(prompt, image_url, question, mllm_model_path)
            return {
                "intent": intent,
                "answer": response,
                "original_image_url": f"/static/uploads/{filename}",
                "yolo_output_image_url": f"/static/output.jpg",
                "yolo_empty": yolo_empty
                    }
        elif intent == "image_defect" and len(yolo_boxes) == 0:
            prompt = '你是一个面向3D打印场景的目标检测大师，具备精准识别、定位图像中固定位置图像是否包含缺陷的能力。\n 职能:从给出的重点区域分析，该区域是否包含异常。\n 该图片没有出现异常或问题，属于正常图片。'
            response = Qwen_image_chat(prompt, image_url, question, mllm_model_path)
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


