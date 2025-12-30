import os
import json
import re
NumOfQuestion = {}

classification_rules = [
    {
        "name": "阶段识别问题",
        "keywords": [
            "属于.*阶段", "铺粉还是打印", "哪个阶段", "哪一步骤", "具体阶段","包含哪些阶段",
            "铺粉环节还是打印环节", "铺粉错误还是打印错误", "铺粉步骤还是打印步骤",
            "铺粉异常还是打印异常", "涂粉还是打印", "粉末涂布还是打印", "印刷阶段还是粉末",
            "喷粉工序", "粉末覆盖阶段", "涂粉环节", "粉末涂布阶段", "铺粉阶段", "打印阶段",
            "涂抹粉末阶段", "喷粉阶段", "上粉阶段", "撒粉阶段", "粉末敷设阶段", "粉末铺设阶段"
        ],
        "regex": [
            r"(属于|处于|是).*(阶段|步骤|环节|工序)",
            r"(铺粉|涂粉|粉末|喷粉|上粉|撒粉).*(还是|或|或者).*(打印|印刷)",
            r"(打印|印刷).*(还是|或|或者).*(铺粉|涂粉|粉末|喷粉|上粉|撒粉)",
            r"哪个.*(阶段|步骤|环节)",
            r"增材制造.*(阶段|步骤|环节)"
        ]
    },
{
        "name": "异常原因分析",
        "keywords": [
            "异常可能原因", "为什么发生异常", "技术问题导致异常","原因","成因","为何","引发","产生的机制","关键因素",
            "为什么发生污染", "为什么铺粉不完全", "为什么球化",
            "为什么翘曲凸起", "导致异常", "形成原因", "技术问题",
            "工艺条件", "设备问题", "软件问题", "参数问题","可能因素", "可能性来源", "引发因素", "引起原因",
            "导致原因", "为何出现", "因素引起", "什么引起",
            "球化原因", "翘曲原因", "铺粉不完整原因"
        ],
        "regex": [
            r"(原因|为什么|为何|如何导致).*(异常|问题)",
            r"导致.*(异常|问题).*(技术|工艺|设备|软件|参数)",
            r"分析.*(异常|问题)",
            r"解决.*(异常|问题)",
            r"异常.*(导致|原因|形成)",
r"(导致|引起|引发).*异常.*(因素|原因)",
            r"异常.*(可能|潜在).*(因素|原因|来源)",
            r"(什么|哪些).*因素.*引起.*异常",
            r"(球化|翘曲|凸起|铺粉不).*(原因|因素|为何)", "技术.*问题"
        ]
    },
    {
        "name": "异常检测问题",
        "keywords": [
            "发生了什么异常", "是否显示异常", "是否发生异常", "哪些位置可能异常",
            "框出问题部分", "不正常的情况", "不寻常的情况", "异常情况", "异常现象",
            "不符合正常情况", "质量问题", "故障", "错误", "缺陷", "瑕疵", "问题区域",
            "潜在异常", "意外瑕疵", "图像失真", "打印异常", "操作异常","非常规状况", "不寻常的事情", "不寻常之处", "不寻常的现象",
            "异样", "异常状况", "异常问题", "问题性质", "需要注意的问题",
            "铺粉不均匀", "铺粉遗漏", "铺粉不完全", "翘曲", "凸起", "球化现象"
        ],
        "regex": [
            r"(什么|哪些).*(异常|问题|情况|现象)",
            r"是否.*(异常|问题|故障|错误)",
            r"(识别|观察|检测|发现).*(异常|问题)",
            r"(框出|标记|指出|标示).*(问题|异常)",
            r"(出现|存在|显示|观察).*(非常规|不寻常|异常|异样)",
            r"(描述|指明|说明).*(不寻常|异常|问题)",
            r"图片.*(是否|有无).*(翘曲|凸起|球化|铺粉不)",
            r"(激光熔融|打印过程).*(球化|翘曲|凸起)",r"是否发生.*事件"
        ]
    },
    {
        "name": "特定异常确认",
        "keywords": [
            "是否发生污染", "是否发生刮痕", "是否发生球化", "是否铺粉不完全",
            "是否翘曲凸起", "污染物异常", "刮痕异常", "球化异常", "铺粉不完全",
            "翘曲凸起", "层厚不均匀", "激光路径错误", "粉末分布异常", "铺粉层不均匀",
            "表面损伤", "图层异常", "颜色渗透", "打印层偏移","球化现象", "翘曲现象", "凸起现象", "铺粉不均匀",
            "铺粉遗漏", "铺粉不完全"
        ],
        "regex": [
            r"是否.*(污染|刮痕|球化|铺粉不完全|翘曲凸起)",
            r"有无.*(污染|刮痕|球化|铺粉不完全|翘曲凸起)",
            r"是否存在.*(污染|刮痕|球化|铺粉不完全|翘曲凸起)",
r"(是否|有无).*(球化|翘曲|凸起|铺粉不)",
            r"(图片|图像).*(存在|显示).*(球化|翘曲|铺粉不)",
            r"是否.*不完全.*现象"
        ]
    },
    {
        "name": "通用异常询问",
        "keywords": [
            "可能会检测到什么异常", "可能会发现哪些问题", "常见异常",
            "潜在问题", "常见故障", "技术挑战", "工艺问题"
        ],
        "regex": [
            r"可能.*(异常|问题|故障)",
            r"常见.*(异常|问题|故障)",
            r"哪些.*(异常|问题|故障)",
            r"需要.*注意"
        ]
    }

]

def classify_question(question):
    """
    根据问题文本分类问题
    """
    for category in classification_rules:
        # 检查关键词匹配
        for keyword in category["keywords"]:
            if keyword in question:
                return category["name"]

        # 检查正则表达式匹配
        for pattern in category["regex"]:
            if re.search(pattern, question):
                return category["name"]

    return "未分类"


def classify_question_by_index(index):
    """
    通过问题的索引编号对问题进行分类

    参数:
        index (tuple): 问题的索引编号，例如 (0, 1) 表示第一组的第二个问题

    返回:
        str: 问题类别
    """
    group_idx = index

    if group_idx == 0:
        return "阶段识别问题"
    elif group_idx == 1:
        return "异常检测问题"
    elif group_idx == 2:
        return "特定异常确认"
    elif group_idx == 3:
        return "通用异常询问"
    elif group_idx == 4:
        return "异常原因分析"
    else:
        return "未知类别"



def classify_question_by_index_error(index):
    """
    通过问题的索引编号对问题进行分类

    参数:
        index (tuple): 问题的索引编号，例如 (0, 1) 表示第一组的第二个问题

    返回:
        str: 问题类别
    """
    group_idx = index

    if group_idx == 0:
        return "异常检测问题"
    elif group_idx == 1:
        return "特定异常确认"
    elif group_idx == 2:
        return "通用异常询问"
    elif group_idx == 3:
        return "异常原因分析"
    else:
        return "未知类别"


# 问题类别说明:
# 1. 阶段识别问题 - 询问图片属于哪个制造阶段
# 2. 异常检测问题 - 询问是否存在异常或异常位置
# 3. 特定异常确认 - 确认是否存在某种特定类型的异常
# 4. 通用异常询问 - 询问某阶段可能出现的通用异常
# 5. 异常原因分析 - 询问异常产生的原因

def classify_question_by_index_withoutimage(index):
    """
    通过问题的索引编号对问题进行分类

    参数:
        index (tuple): 问题的索引编号，例如 (0, 1) 表示第一组的第二个问题

    返回:
        str: 问题类别
    """
    group_idx = index

    if group_idx in [0,1,2,3,4]:
        return "通用异常询问"
    elif group_idx in [5]:
        return "阶段识别问题"
    else:
        return "异常原因分析"


def main():
    if os.path.exists("labels/corrective.json"):
        with open("labels/corrective.json", "r", encoding="utf-8") as f:
            old_data = json.load(f)
            print(len(old_data))

    for item in old_data:
        message = item.get("messages")
        for chat in message:
            if chat["role"] == "user":
                content = chat["content"]
                category = classify_question(content)
                NumOfQuestion[category] = NumOfQuestion.get(category,0) + 1
                chat["category"] = category


                if category =="未分类":
                    print(f"分类:{category}------问题:{content}")
    print(NumOfQuestion)


    # output_file = "/nas_data/taoss/Multi_model_label/labels/withoutImage_category.json"
    # with open(output_file, "w", encoding="utf-8") as f:
    #     json.dump(old_data, f, ensure_ascii=False, indent=2)
    # print(f"\n分类后的数据已保存到 {output_file}")

if __name__ =="__main__":
    main()



