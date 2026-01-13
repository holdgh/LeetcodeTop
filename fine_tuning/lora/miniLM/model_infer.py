import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification


def print_info(msg: str):
    print(f"{8 * '='}{msg}{8 * '='}")


# 推理示例（预测新文本的情感）
def predict(text, tokenizer, model):
    # 预处理文本
    inputs = tokenizer(
        text,
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    ).to("cpu")  # 强制CPU（无GPU时）

    # 推理（关闭梯度计算，提升速度）
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()

    # 映射标签
    label_map = {0: "负向", 1: "正向"}
    return label_map[prediction]


if __name__ == '__main__':
    # model_path = r"C:\Users\gaohu\aiModel\all-MiniLM-L6-v2"
    # model_path = r"C:\Users\gaohu\aiPyProject\LeetcodeTop\fine_tuning\lora\miniLM\best-model-20260106163605"  # 微调版【效果较好，但推理速度较慢】
    # model_path = r"C:\Users\gaohu\aiPyProject\LeetcodeTop\model_smaller\miniLM\minilm_pruned_quant"  # 剪枝/量化版【效果一般】
    model_path = r"C:\Users\gaohu\aiPyProject\LeetcodeTop\model_smaller\miniLM\minilm_pruned_quant_202601061956"  # 剪枝/蒸馏补偿/量化版【效果较好】
    # 1. 加载Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path, trust_remote_code=True)
    print_info("加载分词器完毕")

    # 加载模型：num_labels=2（二分类），若多分类则修改为对应类别数
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     pretrained_model_name_or_path=model_path,
    #     num_labels=2,  # 情感分类为二分类，根据你的任务调整
    #     problem_type="classification",  # 指定任务类型
    #     device_map="auto"  # 自动分配设备（CPU/GPU）
    # )
    # 1. 加载模型时增加dropout，抑制过拟合
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_path,
        num_labels=2,
        # problem_type="classification",
        problem_type="single_label_classification",  # 合法值：单标签分类
        # device_map={"": DEVICE},
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
        # # 新增：添加dropout层，降低过拟合
        # hidden_dropout_prob=0.2,  # 隐藏层dropout
        # attention_probs_dropout_prob=0.2,  # 注意力层dropout
    )
    model.eval()  # 推理模式
    # 测试推理
    test_text1 = "这家酒店环境太差了，卫生不干净，服务也很敷衍"
    test_text2 = "酒店位置很好，房间宽敞明亮，服务人员很热情"
    print(f"文本1预测结果：{predict(text=test_text1, tokenizer=tokenizer, model=model)}")  # 应输出：负向
    print(f"文本2预测结果：{predict(text=test_text2, tokenizer=tokenizer, model=model)}")  # 应输出：正向
