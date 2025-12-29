from openai import OpenAI
import os
question = '你是谁?'

INTENT_PROMPT_TEMPLATE = """
你是一个工业智能问答系统中的意图识别模块，请你根据用户输入的问题，判断其意图属于以下三类之一：

1. defect：关于图片中是否有异常、缺陷、图片处于哪个阶段、缺陷产生的原因等质检相关的问题；
2. model：关于你是谁、你是什么模型、你是哪个公司的等模型身份类问题；
3. other：和质检无关的内容，例如生活、娱乐等内容。

请你直接输出标签（defect/model/other），不要输出其他内容。

问题：{question}
答案：
"""
prompt = INTENT_PROMPT_TEMPLATE.format(question=question)
print(prompt)


# 处理用户问题

# def analyze():
#     intent = ask_intent_llm(question)
#     if intent == "model":
#         return {"answer": "我是上海飞机制造有限公司的天眼多模态质检大模型"}
#     elif intent == "defect":
#         # 走已有图像问答逻辑（你自己的）
#         return run_defect_pipeline(image, question)
#     else:
#         return {"answer": "我是增材质检缺陷检测大模型，您问的问题我暂时无法回答。"}
# aa=analyze()
# print(aa)




# 意图识别提示词模板
INTENT_PROMPT_TEMPLATE = """
你是一个工业智能问答系统中的意图识别模块，请你根据用户输入的问题，判断其意图属于以下三类之一：

1. defect：关于图片中是否有异常、缺陷、图片处于哪个阶段、缺陷产生的原因等质检相关的问题；
2. model：关于你是谁、你是什么模型、你是哪个公司的等模型身份类问题；
3. other：和质检无关的内容，例如生活、娱乐等内容。

请你直接输出标签（defect/model/other），不要输出其他内容。

问题：{question}
答案：
"""
def classify_question_intent(question: str) -> str:
    try:
        client = OpenAI(
            api_key="sk",
            # base_url="https://api.fe8.cn/v1"  # gpt
            # base_url='http://localhost:8000/v1'  # 本地启用模型
            base_url='http://192.168.0.213:6662/v1'

        )
        
        prompt = INTENT_PROMPT_TEMPLATE.format(question=question)
        response = client.chat.completions.create(
            # model="gpt-4",
            # model='/nas_data/models/Qwen/Qwen3-32B/',
            # model='Qwen3-32B',
            # model='Qwen2.5-7B-Instruct',
            model='Qwen3-4B',

            messages=[{"role": "user", "content": prompt}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}, # 禁止qwen输出<think>部分
            temperature=0.7
        )
        result = response.choices[0].message.content.strip().lower()
        return result
    except Exception as e:
        raise RuntimeError(f"本地模型调用失败：{str(e)}")
    

# 示例问题测试
if __name__ == "__main__":
    questions = [
        "这张图有没有缺陷？",
        "你是谁？",
        "你喜欢看电影吗？",
        "这个缺陷是哪个阶段产生的？",
        "你是哪个公司的？"
    ]

    for q in questions:
        intent = classify_question_intent(q)
        print(f"问题：{q}\n识别意图：{intent}\n")