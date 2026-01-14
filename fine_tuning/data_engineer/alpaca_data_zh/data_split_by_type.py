import json
import random
import pandas as pd
from collections import defaultdict
import datetime

file_suffix = datetime.datetime.now().strftime(format="%Y%m%d%H%M%S")

# ====================== 配置项 ======================
RAW_DATA_PATH = "../../alpaca_zh/alpaca_data_zh_51k.json"  # 原始数据集路径
STRATIFIED_DATA_PATH = "./alpaca_zh_stratified"  # 分层后数据集保存路径
TRAIN_RATIO = 0.95  # 训练集比例
SEED = 42  # 固定随机种子，保证结果可复现


# ====================== 步骤1：定义业务类型规则 ======================
def label_business_type(instruction: str) -> str:
    """
    规则匹配：给每条指令打业务类型标签
    可根据实际需求扩展规则
    """
    instruction = instruction.lower().strip()

    # 1. 总结归纳类（包含“总结”“归纳”“概括”“摘要”等关键词）
    if any(key in instruction for key in ["总结", "归纳", "概括", "摘要", "整理"]):
        return "总结归纳"

    # 2. 算法推理类（包含“计算”“推理”“解题”“算”“证明”等关键词）
    elif any(key in instruction for key in ["计算", "推理", "解题", "算", "证明", "推导", "公式"]):
        return "算法推理"

    # 3. 知识问答类（包含“是什么”“为什么”“怎么样”“多少”“哪里”等）
    elif any(key in instruction for key in ["是什么", "为什么", "怎么样", "多少", "哪里", "谁", "何时", "如何"]):
        return "知识问答"

    # 4. 创意生成类（包含“写”“创作”“生成”“编”“设计”等）
    elif any(key in instruction for key in ["写", "创作", "生成", "编", "设计", "构思", "仿写"]):
        return "创意生成"

    # 5. 其他类型
    else:
        return "其他"


# ====================== 步骤2：加载数据并打标签 ======================
def load_and_label_data():
    # 加载原始数据
    with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # 打标签
    labeled_data = []
    type_count = defaultdict(int)  # 统计各类型数量
    for idx, sample in enumerate(raw_data):
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        output_text = sample.get("output", "")

        # 合并指令+输入，提升标签准确性
        full_text = instruction + " " + input_text
        business_type = label_business_type(full_text)

        labeled_sample = {
            "id": idx,
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
            "business_type": business_type
        }
        labeled_data.append(labeled_sample)
        type_count[business_type] += 1

    # 打印类型分布
    print("===== 原始数据业务类型分布 =====")
    total = len(labeled_data)
    for type_name, count in type_count.items():
        print(f"{type_name}: {count}条（{count / total * 100:.2f}%）")

    return labeled_data, type_count


# ====================== 步骤3：分层抽样 ======================
def stratified_split(labeled_data):
    # 按业务类型分组
    type_to_samples = defaultdict(list)
    for sample in labeled_data:
        type_to_samples[sample["business_type"]].append(sample)

    train_data = []
    val_data = []
    # 临时增加
    else_data = []

    # 对每个类型单独划分训练/验证集
    for type_name, samples in type_to_samples.items():
        random.seed(SEED)
        random.shuffle(samples)  # 打乱该类型样本

        # 计算该类型的训练集数量
        train_num = int(len(samples) * TRAIN_RATIO)

        # 划分
        train_samples = samples[:train_num]
        val_samples = samples[train_num:]

        train_data.extend(train_samples)
        val_data.extend(val_samples)

        # 打印该类型的划分结果
        print(f"\n===== {type_name} 划分结果 =====")
        print(f"训练集：{len(train_samples)}条，验证集：{len(val_samples)}条")
        # 临时增加
        if type_name == "其他":
            else_data = samples

    # 打乱训练集/验证集（避免同类型样本扎堆）
    random.seed(SEED)
    random.shuffle(train_data)
    random.shuffle(val_data)

    # 验证整体划分比例
    print(f"\n===== 整体划分结果 =====")
    print(f"训练集总数：{len(train_data)}条（{len(train_data) / (len(train_data) + len(val_data)) * 100:.2f}%）")
    print(f"验证集总数：{len(val_data)}条（{len(val_data) / (len(train_data) + len(val_data)) * 100:.2f}%）")

    return train_data, val_data, else_data


# ====================== 步骤4：保存分层后的数据集 ======================
def save_stratified_data(train_data, val_data, else_data):
    # 创建保存目录
    import os
    if not os.path.exists(STRATIFIED_DATA_PATH):
        os.makedirs(STRATIFIED_DATA_PATH)

    # 保存训练集/验证集（保留原始格式，仅新增business_type字段）
    train_save_path = os.path.join(STRATIFIED_DATA_PATH, f"train_{file_suffix}.json")
    val_save_path = os.path.join(STRATIFIED_DATA_PATH, f"val_{file_suffix}.json")
    # 临时增加
    else_save_path = os.path.join(STRATIFIED_DATA_PATH, f"else_{file_suffix}.json")

    # 移除id字段（可选，不影响微调）
    train_data_clean = [{k: v for k, v in s.items() if k != "id"} for s in train_data]
    val_data_clean = [{k: v for k, v in s.items() if k != "id"} for s in val_data]
    # 临时增加
    else_data_clean = [{k: v for k, v in s.items() if k != "id"} for s in else_data]

    with open(train_save_path, "w", encoding="utf-8") as f:
        json.dump(train_data_clean, f, ensure_ascii=False, indent=2)

    with open(val_save_path, "w", encoding="utf-8") as f:
        json.dump(val_data_clean, f, ensure_ascii=False, indent=2)
    # 临时增加
    with open(else_save_path, "w", encoding="utf-8") as f:
        json.dump(else_data_clean, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 分层后的数据集已保存至：{STRATIFIED_DATA_PATH}")
    print(f"训练集路径：{train_save_path}")
    print(f"验证集路径：{val_save_path}")


# ====================== 主函数 ======================
if __name__ == "__main__":
    # 步骤1：加载数据并打标签
    labeled_data, type_count = load_and_label_data()

    # 步骤2：分层抽样
    train_data, val_data, else_data = stratified_split(labeled_data)

    # 步骤3：保存数据
    save_stratified_data(train_data, val_data, else_data)

    # （可选）统计分层后训练/验证集的类型分布
    print("\n===== 训练集类型分布 =====")
    train_type_count = defaultdict(int)
    for sample in train_data:
        train_type_count[sample["business_type"]] += 1
    for type_name, count in train_type_count.items():
        print(f"{type_name}: {count}条（{count / len(train_data) * 100:.2f}%）")

    print("\n===== 验证集类型分布 =====")
    val_type_count = defaultdict(int)
    for sample in val_data:
        val_type_count[sample["business_type"]] += 1
    for type_name, count in val_type_count.items():
        print(f"{type_name}: {count}条（{count / len(val_data) * 100:.2f}%）")