
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
from sf_question import (classify_question_intent, reprocessing, qwen3_AM_knowledge)
from yolo_script import (
    image_predict, draw_yolo_boxes, format_yolo_detections, InterVL_image_chat, InterVL_chat, Qwen_image_chat, Qwen_chat,
    YOLO
)

app = FastAPI()
os.makedirs("/nas_data/lyl_project/sf/static/uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="/nas_data/lyl_project/sf/static"), name="static")
templates = Jinja2Templates(directory="templates")
yolo_model = YOLO('/home/liuyl/sf/test/project/best.pt')
class_names = ["刮刀横/竖条纹", "曲翘凸起", "污染物", "污染物", "球化", "刮刀横/竖条纹", "刮刀横/竖条纹", "曲翘凸起", "曲翘凸起","铺粉不完全"]
# mllm_model_path = '/home/liuyl/llama_factory/LLaMA-Factory/output/qwen2.5_vl_lora_sft'
# mllm_model_path = '/nas_data/lyl_project/llamafactory/LLaMA-Factory/output/qwen2.5_vl_7b_4data_lora_sft'
mllm_model_path ='/nas_data/lyl_project/llamafactory_medium_model/saves/qwen2.5_vl-7b/213/sf_4data_num_3wdata/full/sft'
# mllm_model_path ='/nas_data/lyl_project/llamafactory_medium_model/saves/qwen2.5_vl-3b/129/sf_4data_num_3wdata/full/sft/checkpoint-1293'
base_url="http://192.168.0.194:6660/v1"

# mllm_model_path ='InterVL2B'
# base_url="http://127.0.0.1:8000/v1"

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index_new2.html", {"request": request})

@app.post("/analyze/")
async def analyze(question: str = Form(...),
                  image: Optional[UploadFile] = File(None)
                  ):
    original_image_url = None
    yolo_output_image_url = None
    yolo_empty = True

    if image is None:
        intent = classify_question_intent(question)
        if intent == "model":
            response = "我是“上海联通”和“上飞公司航研所”共同打造的的“灵镜”多模态质检大模型"
            
        elif intent == "image_defect":
            response = "请上传图片"

        elif intent == "defect":
            prompt = "你是一个3D打印增材质检AI助手。\n 职能:按照要求详细回答用户咨询的问题。"
            # response = InterVL_chat(base_url, prompt, question, mllm_model_path)
            # response = qwen3_AM_knowledge(question)
            print('base_url',base_url)
            response = Qwen_chat(base_url, prompt, question, mllm_model_path)

        else:
            response = "抱歉，我是为增材制造质检量身打造的多模态AI助手，您问的问题我暂时无法回答。"
 
    else:
        filename = f"{uuid.uuid4()}.jpg"
        filepath = f"/nas_data/lyl_project/sf/static/uploads/{filename}"
        
        original_image_url = f"static/uploads/{filename}"
        with open(filepath, "wb") as f:
            f.write(await image.read())
        image_url = os.path.abspath(filepath)
        output_path = "/nas_data/lyl_project/sf/static/output.jpg"
        yolo_boxes = image_predict(yolo_model, image_url)
        draw_yolo_boxes(image_url, yolo_boxes, output_path, class_names)
        yolo_empty = len(yolo_boxes) == 0
        print('yolo_empty',yolo_empty)
        intent = classify_question_intent(question)
        print('intent',intent)
        if intent == "model":
            response = "我是“上海联通”和“上飞公司航研所”共同打造的的“灵镜”多模态质检大模型"
            original_image_url = None
            print('aaa', f"/nas_data/lyl_project/sf/static/uploads/{filename}")
            
        elif intent == "defect":
            prompt = "你是一个3D打印增材质检AI助手。\n 职能:按照要求详细回答用户咨询的问题。"
            # response = InterVL_chat(base_url, prompt, question, mllm_model_path)
            # response = qwen3_AM_knowledge(question)
            response = Qwen_chat(base_url, prompt, question, mllm_model_path)

        elif intent == "image_defect" and len(yolo_boxes) != 0:
            yolo_output_image_url = f"/static/output.jpg"
            prompt = format_yolo_detections(image_url, yolo_boxes, class_names)
            # response = InterVL_image_chat(base_url, prompt, image_url, question, mllm_model_path)
            response = Qwen_image_chat(base_url, prompt, image_url, question, mllm_model_path)
            print('mllm回答：', response)
            # response = reprocessing(response)
            # print('--------------------------')
            # print('后处理结果：', response)

        elif intent == "image_defect" and len(yolo_boxes) == 0:
            yolo_output_image_url = f"/static/output.jpg"
            prompt = '你是一个面向3D打印场景的目标检测大师，具备精准分析图像缺陷的能力。\n 职能:从给出的重点区域分析，该区域是否包含异常。\n 该图片没有出现异常或问题，属于正常图片。'
            # response = InterVL_image_chat(base_url, prompt, image_url, question, mllm_model_path)
            response = Qwen_image_chat(base_url, prompt, image_url, question, mllm_model_path)

        else:
            response = "抱歉，我是为增材制造质检量身打造的多模态AI助手，您问的问题我暂时无法回答。"
            original_image_url = None
       
    return {
            "intent": intent,
            "answer": response,
            "original_image_url": original_image_url,
            "yolo_output_image_url": yolo_output_image_url,
            "yolo_empty": yolo_empty
                }


