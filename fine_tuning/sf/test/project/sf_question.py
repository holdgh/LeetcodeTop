from openai import OpenAI
import os

# # 意图识别用户问题
# def question_analyze(question):
#     intent = classify_question_intent(question)
#     if intent == "model":
#         return {"answer": "我是上海飞机制造有限公司的灵镜多模态质检大模型"}
#     elif intent == "defect":
#         # return run_defect_pipeline(image, question)
#         return '走已有图像问答逻辑'
#     else:
#         return {"answer": "我是增材质检缺陷检测大模型，您问的问题我暂时无法回答。"}

def classify_question_intent(question: str) -> str:
    # 意图识别提示词模板
    INTENT_PROMPT_TEMPLATE = """
    你是一个工业智能问答系统中的意图识别模块，请你根据用户输入的问题，判断其意图属于以下四类之一：

    1. image_defect：关于图片中是否有异常、问题、缺陷、图片处于哪个阶段、图片中缺陷产生的原因等质检相关的问题；
    2. defect：询问关于增材质检的通用知识，例如增材质检中铺粉或打印阶段会发生什么缺陷，某缺陷发生的原因等问题；
    3. model：关于你是谁、你是什么模型、你是哪个公司的等模型身份类问题；
    4. other：和质检无关的内容，例如生活、娱乐等内容。
    请你直接输出标签（defect/model/other），不要输出其他内容。

    问题：{question}
    答案：
    """
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
            model='Qwen3-4B',
            messages=[{"role": "user", "content": prompt}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}, # 禁止qwen输出<think>部分
            temperature=0.7
        )
        result = response.choices[0].message.content.strip().lower()
        return result
    except Exception as e:
        raise RuntimeError(f"本地模型调用失败：{str(e)}")



def reprocessing(question: str) -> str:
    # 意图识别提示词模板
    INTENT_PROMPT_TEMPLATE = """
    任务要求：请对多模态大模型的回答进行后处理，删除所有关于缺陷在图片上的位置描述（如 “边缘位置”“中部区域”“中间和上侧” 等具体方位表述），其他内容（缺陷类型、特征描述、分析逻辑）保持不变，不新增任何信息。
    如果多模态大模型的回答中没有位置描述，则直接输出回答。

    输入示例：
    图片中一共出现 3 种缺陷问题。
    第一种缺陷类型是曲翘凸起，主要出现在零件的边缘位置。该区域中可以看到零件的边缘轮廓，呈现白色或金属光泽，没有铺上金属粉末。因此可以分析在这些位置发生了曲翘凸起异常。
    第二种缺陷类型是污染物，出现在图片的中部区域。该位置区域中可以看到明显的深灰色或黑色阴影。因此可以分析在这个位置发生了污染物异常。
    第三种缺陷类型是刮刀横 / 竖条纹，贯穿于图片的中间和上侧。该位置区域展现出深黑色的横线或竖线。因此可以分析在这些位置发生了刮刀横 / 竖条纹异常。

    输出要求示例：
    图片中一共出现 3 种缺陷问题。
    第一种缺陷类型是曲翘凸起。可以看到零件的边缘轮廓，呈现白色或金属光泽，没有铺上金属粉末。因此可以分析发生了曲翘凸起异常。
    第二种缺陷类型是污染物。可以看到明显的深灰色或黑色阴影。因此可以分析发生了污染物异常。
    第三种缺陷类型是刮刀横/竖条纹。展现出深黑色的横线或竖线。因此可以分析发生了刮刀横/竖条纹异常。

    多模态大模型的回答：{question}
    
    """
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
            model='Qwen3-4B',
            messages=[{"role": "user", "content": prompt}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}, # 禁止qwen输出<think>部分
            temperature=0.9
        )
        result = response.choices[0].message.content.strip().lower()
        
        return result
    except Exception as e:
        raise RuntimeError(f"本地模型调用失败：{str(e)}")


def qwen3_AM_knowledge(question: str) -> str:
    # INTENT_PROMPT_TEMPLATE = """
    # 【角色】你是一个面向3D打印增材质检场景下的通用知识大师，具备回答增材质检过程中的相关问题的能力。\n 职能:按照要求详细回答用户咨询的问题。
    # # 增材质检过程分为铺粉阶段和打印阶段，先铺一层粉，激光进行打印，然后再铺粉多次循环。铺粉阶段和打印阶段都会出现问题异常或缺陷。铺粉阶段会出现的异常有：铺粉不完全、刮刀产生横/竖条纹；打印阶段会出现的异常有：翘曲凸起、污染物、球化。
    # # 缺陷类型以及缺陷对应的特征描述
    # ## 铺粉不完全：正常铺粉图片应该表现为均匀的、灰色的。但是在图片中出现大面积的、在零件表面展示的白色亮面。主要是由于粉末供应不足、风机转速过高把粉吹走，导致该铺上粉的地方没铺上。
    #     铺粉不完全（Incomplete Powder Spreading）是影响打印质量的关键问题之一。它会导致局部粉末缺失、层厚不均，进而引发未熔合、孔隙、表面粗糙等缺陷。
    #     铺粉不完全表现在局部粉末缺失，某些区域无粉末覆盖，露出下层已固化材料或基板。
    # ## 曲翘凸起：正常铺粉图片应该表现为均匀的、灰色的。
    #     翘起凸起发生在打印阶段，但是在铺粉阶段更容易被发现，在铺粉的图片中能看到零件的边缘轮廓，呈现白色或金属光泽，则表示零件边缘没有铺上金属粉末。
    #     这类异常都是打印时过熔产生的，一般来说在边缘轮廓位置，激光行进路程较短，能量较为集中更容易产生翘曲。只要是高于粉床表面的统一认为是翘曲凸起。
    #     激光熔融3D打印过程中出现的翘曲凸起是一个常见的工艺缺陷，主要由材料在快速加热和冷却过程中产生的不均匀热应力或者打印功率较高扫描速度较慢引起。
    # ## 刮刀横/竖条纹：正常铺粉图片应该表现为均匀的、灰色的。
    #     “刮刀横/竖条纹”异常的明显特征是能在图片中看到贯穿图片的深黑色的横线，从图片左侧一直到右侧。
    #     刮刀上有缺口、颗粒会在铺粉时产生一个横条纹，刮刀碰到凸起时抖动可能会在铺粉时产生竖条纹。
    #     刮刀磨损或变形：长期使用后，刮刀边缘可能出现磨损、缺口或弯曲，导致铺粉时形成不均匀的条纹。
    #     刮刀与粉床间隙不均：若刮刀安装不平整或Z轴运动误差，可能导致部分区域铺粉过厚或过薄，形成横向条纹。
    #     铺粉速度过快：刮刀移动速度过高时，粉末可能被推挤而非均匀铺展，导致条纹状分布。
    # ## 球化：在打印阶段，过熔会导致金属表面形貌出现异常，打印后的金属面平整度出现异常。在铺粉阶段时，图片中表现的特征零件表面粗糙部分位置可以被粉末覆盖，但是其内部会有孔隙，表现为白色的亮点和黑色的球状颗粒，因此在铺粉阶段同样能发现球化的异常。
    #     球化是指熔融金属在冷却过程中未能均匀铺展，反而收缩成球状或椭球状的现象。这一缺陷会严重影响打印件的致密度、表面质量和机械性能。
    #     球化的金属表面粗糙，打印件表面布满凸起的球状或椭球状颗粒。孔隙率高，球体之间未完全结合，形成蜂窝状孔隙。
    # ## 污染物：灰渣/残余颗粒/飞溅颗粒统一认为是污染物，从图像特征上看，这些污染物表现特征是明显的深灰色或黑色阴影。
    #     污染物形成的原因可能是打印过程中激光温度不够高，金属粉末未完全被熔融遗留的残灰。也可能是激光温度过高导致过熔后碎片飞溅形成的。
    # 问题：{question}
    # """
    INTENT_PROMPT_TEMPLATE = """
        【角色】你是一位专注于3D打印增材质检领域的专家，擅长清晰阐释增材质检过程中的技术问题和缺陷分析。
        【回答规范】
            **格式要求**：
            - 禁用Markdown符号（如---、**、-等），采用纯文本分段表述
            - 每个异常类型以"### 异常名称"开头，后续内容简洁分点
            - 避免使用冗余过渡句，直接呈现核心信息

        【任务要求】
        1. 针对用户提出的问题，提供准确、专业的回答
        2. 回答需逻辑清晰，避免冗余信息
        3. 不要用到模型自己的关于增材质检的相关知识

        【增材质检基础知识】
        增材质检主要分为两个关键阶段：
        1. 铺粉阶段：粉末层的均匀性直接影响打印质量
        2. 打印阶段：激光熔融过程中易出现热应力相关缺陷

        【铺粉阶段常见异常】
        1. 铺粉不完全
        - 特征：零件表面出现大面积白色亮区，对应粉末缺失区域
        - 成因：供粉系统故障、刮刀速度过快或风机气流干扰
        - 影响：导致未熔合、孔隙率增加等后续缺陷

        2. 刮刀横/竖条纹
        - 特征：图像中出现贯穿性黑色条纹（横线多为刮刀磨损，竖线常因刮刀振动）
        - 成因：刮刀边缘磨损、安装不平整或Z轴运动偏差
        - 影响：造成层厚不均，影响打印精度和表面质量

        【打印阶段常见异常】
        1. 翘曲凸起
        - 特征：零件边缘或局部区域向上翘起，铺粉图像中表现为金属光泽边缘
        - 成因：激光功率过高/扫描速度过慢导致热应力集中
        - 影响：降低零件与基板结合力，可能导致层间脱离

        2. 球化
        - 特征：表面出现球状颗粒和黑色孔隙，伴随粗糙度增加
        - 成因：激光能量密度过高，熔池不稳定
        - 影响：降低零件致密度和力学性能

        3. 污染物
        - 特征：图像中呈现深灰色/黑色阴影（未熔粉末、飞溅物或氧化物）
        - 成因：激光参数波动、粉末质量问题或环境杂质
        - 影响：导致局部缺陷，可能引发应力集中

        问题：{question}
        """
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
            model='Qwen3-4B',
            messages=[{"role": "user", "content": prompt}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}, # 禁止qwen输出<think>部分
            temperature=0.7
        )
        result = response.choices[0].message.content.strip().lower()
        return result
    except Exception as e:
        raise RuntimeError(f"本地模型调用失败：{str(e)}")







# if __name__ == "__main__":
#     questions = [
#         "这张图有没有缺陷？",
#         "你是谁？",
#         "你喜欢看电影吗？",
#         "这个缺陷是哪个阶段产生的？",
#         "你是哪个公司的？"
#     ]
#     for q in questions:
#         intent = classify_question_intent(q)
#         print(f"问题：{q}\n识别意图：{intent}\n")
#         print(question_analyze(q))