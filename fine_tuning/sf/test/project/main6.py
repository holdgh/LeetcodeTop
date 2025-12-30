# 未解决：不上传图片仅问问题
# 已解决：意图识别模块;异常图片调添加nomal微调模型输出
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import uuid
import os
from openai import OpenAI
from sf_question import classify_question_intent, question_analyze
from yolo_script import (
    image_predict, draw_yolo_boxes, format_yolo_detections,
    chat, chat2, YOLO, class_names, mllm_model_path
)

# 初始化 FastAPI 应用和配置
app = FastAPI()
os.makedirs("static/uploads", exist_ok=True)  # 确保上传目录存在
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 初始化 YOLO 模型
yolo_model = YOLO('/home/liuyl/sf/test/best.pt')

# 首页路由
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index_test.html", {"request": request})

# 分析接口
@app.post("/analyze/")
async def analyze(
    question: str = Form(...),  # 必需的问题参数
    image: UploadFile = File(None)  # 可选的图片参数
):
    # 获取问题意图
    intent = classify_question_intent(question)
    print(f"问题参数===>: {question}...图片参数===>：{image.filename}")
    print(f"问题意图: {intent}")
    # 初始化响应字典
    response = {
        "intent": intent,
        "answer": "",
        "original_image_url": None,
        "yolo_output_image_url": None,
        "yolo_empty": True
    }

    # 处理无图片情况
    if not image:
        if intent == "model":
            response["answer"] = "我是 上海联通 和 上飞公司航研所 共同打造的的“灵镜”多模态质检大模型"
        elif intent == "defect":
            response["answer"] = "请上传图片"
        else:
            response["answer"] = "抱歉，我是为增材制造质检量身打造的多模态AI助手，您问的问题我暂时无法回答。"
        return response

    # 处理图片上传
    filename = f"{uuid.uuid4()}.jpg"
    filepath = f"static/uploads/{filename}"
    with open(filepath, "wb") as f:
        f.write(await image.read())

    # 执行目标检测
    image_url = os.path.abspath(filepath)
    yolo_boxes = image_predict(yolo_model, image_url)
    yolo_empty = len(yolo_boxes) == 0

    # 更新响应中的图片信息
    response.update({
        "original_image_url": f"/static/uploads/{filename}",
        "yolo_empty": yolo_empty
    })

    # 根据意图处理响应
    if intent == "model":
        response["answer"] = "我是 上海联通 和 上飞公司航研所 共同打造的的“灵镜”多模态质检大模型"
        return response

    # 处理缺陷检测请求
    if intent == "defect":
        if yolo_empty:
            # 未检测到缺陷时使用微调模型分析
            prompt = "你是一个面向3D打印场景的目标检测大师，具备精准识别、定位图像中固定位置图像是否包含缺陷的能力。\n 职能:从给出的重点区域分析，该区域是否包含异常。"
            response["answer"] = chat(prompt, image_url, question, mllm_model_path)
            return response

        # 检测到缺陷时绘制边界框并分析
        draw_yolo_boxes(image_url, yolo_boxes, "static/output.jpg", class_names)
        prompt = format_yolo_detections(image_url, yolo_boxes, class_names)
        response.update({
            "answer": chat(prompt, image_url, question, mllm_model_path),
            "yolo_output_image_url": f"/static/output.jpg"
        })
        return response

    # 处理其他意图
    response["answer"] = "抱歉，我是为增材制造质检量身打造的多模态AI助手，您问的问题我暂时无法回答。"
    return response







# 未解决：不上传图片仅问问题
# 已解决：意图识别模块;异常图片调添加nomal微调模型输出
# from fastapi import FastAPI, File, UploadFile, Form
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.requests import Request
# from PIL import Image
# import uuid
# import os
# from openai import OpenAI
# import os
# from typing import Optional
# from sf_question import (classify_question_intent,question_analyze)
# from yolo_script import (
#     image_predict, draw_yolo_boxes, format_yolo_detections, chat,chat2,
#     YOLO, class_names, mllm_model_path
# )
#
# app = FastAPI()
# os.makedirs("static/uploads", exist_ok=True)
# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")
#
# yolo_model = YOLO('/home/liuyl/sf/test/best.pt')
#
# @app.get("/", response_class=HTMLResponse)
# async def home(request: Request):
#     return templates.TemplateResponse("index_test.html", {"request": request})
#
# @app.post("/analyze/")
# async def analyze(question: str = Form(...),
#                   image: Optional[UploadFile] = File(None)
#                   ):
#     if image is None:
#         intent = classify_question_intent(question)
#         if intent == "model":
#             response = "我是 上海联通 和 上飞公司航研所 共同打造的的“灵镜”多模态质检大模型"
#
#         elif intent == "defect":
#             response = "请上传图片"
#         else:
#             response = "抱歉，我是为增材制造质检量身打造的多模态AI助手，您问的问题我暂时无法回答。"
#         return {
#         "intent": intent,
#         "answer": response,
#         "original_image_url": None,
#         "yolo_output_image_url": None,
#         "yolo_empty": True
#             }
#
#     else:
#         filename = f"{uuid.uuid4()}.jpg"
#         filepath = f"static/uploads/{filename}"
#         with open(filepath, "wb") as f:
#             f.write(await image.read())
#         image_url = os.path.abspath(filepath)
#         output_path = "static/output.jpg"
#         yolo_boxes = image_predict(yolo_model, image_url)
#         draw_yolo_boxes(image_url, yolo_boxes, output_path, class_names)
#         yolo_empty = len(yolo_boxes) == 0
#         print('yolo_empty',yolo_empty)
#
#         intent = classify_question_intent(question)
#         print('intent',intent)
#         if intent == "model":
#             response = "我是 上海联通 和 上飞公司航研所 共同打造的的“灵镜”多模态质检大模型"
#             return {
#                 "intent": intent,
#                 "answer": response,
#                 "original_image_url": f"/static/uploads/{filename}",
#                 "yolo_output_image_url": None,
#                 "yolo_empty": yolo_empty
#                     }
#         elif intent == "defect" and len(yolo_boxes) == 0:
#             prompt = "你是一个面向3D打印场景的目标检测大师，具备精准识别、定位图像中固定位置图像是否包含缺陷的能力。\n 职能:从给出的重点区域分析，该区域是否包含异常。"
#             response = chat(prompt, image_url, question, mllm_model_path)
#             # response = chat2(prompt, question, mllm_model_path)
#
#             return {
#                 "intent": intent,
#                 "answer": response,
#                 "original_image_url": f"/static/uploads/{filename}",
#                 "yolo_output_image_url": None,
#                 "yolo_empty": yolo_empty
#                     }
#         elif intent == "defect" and len(yolo_boxes) != 0:
#             prompt = format_yolo_detections(image_url, yolo_boxes, class_names)
#             response = chat(prompt, image_url, question, mllm_model_path)
#             return {
#                 "intent": intent,
#                 "answer": response,
#                 "original_image_url": f"/static/uploads/{filename}",
#                 "yolo_output_image_url": f"/static/output.jpg",
#                 "yolo_empty": yolo_empty
#                     }
#         else:
#             response = "抱歉，我是为增材制造质检量身打造的多模态AI助手，您问的问题我暂时无法回答。"
#             return {
#                 "intent": intent,
#                 "answer": response,
#                 "original_image_url": f"/static/uploads/{filename}",
#                 "yolo_output_image_url": None,
#                 "yolo_empty": yolo_empty
#                     }
#

