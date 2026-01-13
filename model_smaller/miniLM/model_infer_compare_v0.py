import os
import time

import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

VAL_DATA_PATH = r'C:\Users\gaohu\aiPyProject\LeetcodeTop\fine_tuning\jd_comment\cleaned_dataset.csv'
DEVICE = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")


def print_info(msg: str):
    print(f"{8 * '='}{msg}{8 * '='}")


# ====================== 新增：分层随机抽样函数 ======================
def stratified_sample_data(data_path, sample_num=1000, random_state=42):
    """
    分层随机抽样，保证评估集标签分布与原始数据一致
    :param data_path: 原始数据路径
    :param sample_num: 抽样数量
    :param random_state: 随机种子（保证结果可复现）
    :return: 抽样后的texts, labels
    """
    # 1. 加载原始数据
    df = pd.read_csv(data_path, encoding="utf-8")

    df = df.drop(labels='dataset', axis=1)  # 仅保留文本及类别标签列
    df = df.dropna()  # 删除缺省值的行
    # 过滤无效数据
    df = df[df["sentence"].notna() & (df["sentence"].str.len() > 5)].reset_index(drop=True)

    # 2. 检查抽样数量是否合理
    # if sample_num > len(df):
    #     print(f"⚠️  抽样数量({sample_num})超过数据总量({len(df)})，使用全部数据")
    #     sample_df = df.copy()
    # else:
    #     # 3. 初始化分层抽样器（仅生成1组索引）
    #     sss = StratifiedShuffleSplit(
    #         n_splits=1,  # 仅1组抽样结果
    #         test_size=sample_num,  # 直接指定抽样数量（替代比例）
    #         random_state=random_state  # 固定随机种子
    #     )
    #     # 4. 直接提取抽样索引（无循环，消除歧义）
    #     # split返回迭代器，转为列表后取第0组的测试索引
    #     test_indices = list(sss.split(df, df["label"]))[0][1]
    #     # 5. 根据索引抽取样本（显式初始化sample_df）
    #     sample_df = df.iloc[test_indices].reset_index(drop=True)
    if sample_num > len(df):
        print(f"⚠️  抽样数量({sample_num})超过数据总量({len(df)})，使用全部数据")
        sample_num = len(df)
    """
    为什么下述sample_df的循环赋值不会有问题
        StratifiedShuffleSplit(n_splits=1, ...)的核心特性：
            n_splits=1 表示只生成1 组训练 / 测试索引（而非多组）；
            循环for train_idx, test_idx in sss.split(...) 本质上只执行1 次，因此sample_df只会被赋值 1 次，不存在 “取最后一次值” 的问题；
            这种写法是sklearn官方推荐的（兼容n_splits>1的场景），但对于n_splits=1的单抽样场景，确实容易让人误解。
    """
    # 3. 分层随机抽样（按label分层）
    sss = StratifiedShuffleSplit(n_splits=1, test_size=sample_num / len(df), random_state=random_state)
    # 获取抽样索引
    for train_idx, test_idx in sss.split(df["sentence"], df["label"]):
        sample_df = df.iloc[test_idx].reset_index(drop=True)

    # 4. 打印抽样分布，验证一致性
    original_dist = df["label"].value_counts(normalize=True).round(4)
    sample_dist = sample_df["label"].value_counts(normalize=True).round(4)
    print(f"\n📊 样本分布验证：")
    print(f"原始数据分布：{original_dist.to_dict()}")
    print(f"抽样数据分布：{sample_dist.to_dict()}")

    return sample_df["sentence"].tolist(), sample_df["label"].tolist()


def model_infer_batch(model_path: str, data_path: str, sample_num: int, model_flag: str):
    print_info(f"开启{model_flag}批量测试，测试数据条数：{sample_num}")
    # 1. 加载Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path, trust_remote_code=True)
    print_info("加载分词器完毕")
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_path,
        num_labels=2,
        problem_type="single_label_classification",
        trust_remote_code=True,
        ignore_mismatched_sizes=True
    )
    model.eval()  # 推理模式
    # 执行验证
    """验证轻量化后模型的精度和推理速度"""
    # 加载测试数据
    # df = pd.read_csv(data_path, encoding="utf-8")
    # df = df.drop(labels='dataset', axis=1)  # 仅保留文本及类别标签列
    # df = df.dropna()  # 删除缺省值的行
    # df = df[df["sentence"].str.len() > 5].reset_index(drop=True)
    # texts = df["sentence"].tolist()[:sample_num]  # 取100条样本测试
    # labels = df["label"].tolist()[:sample_num]
    texts, labels = stratified_sample_data(data_path, sample_num)
    # 推理速度测试
    start_time = time.time()  # 通用计时起点【CPU/GPU皆可用】
    preds = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                f"判断情感倾向：{text}",
                max_length=128,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(DEVICE)
            # GPU额外计时（可选，更精准）
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()  # 等待之前的GPU操作完成
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            # 模型推理
            output = model(**inputs)
            pred = torch.argmax(output.logits, dim=-1).cpu().item()
            preds.append(pred)
            if DEVICE.type == "cuda":
                end.record()
                torch.cuda.synchronize()
    # 计算总推理时间（通用方案）
    total_time = time.time() - start_time
    avg_time_per_sample = total_time / sample_num  # 单样本平均推理时间

    # 精度计算
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")

    # 模型大小计算
    # ====================== 核心工具函数：准确计算模型大小 ======================
    def get_accurate_model_size(model_dir):
        """
        准确计算模型文件总大小（兼容safetensors/bin/pt/pth/ckpt）
        :param model_dir: 模型文件夹路径
        :return: 模型总大小（MB），保留2位小数
        """
        # 1. 校验路径是否存在
        if not os.path.exists(model_dir):
            print(f"⚠️  模型路径不存在：{model_dir}")
            return 0.0

        # 2. 定义需要统计的模型权重文件后缀
        weight_suffixes = [".safetensors", ".bin", ".pt", ".pth", ".ckpt"]
        total_size = 0

        # 3. 遍历文件夹所有文件
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                # 仅统计权重文件，跳过配置文件（如config.json）
                if any(file.endswith(suffix) for suffix in weight_suffixes):
                    file_path = os.path.join(root, file)
                    # 获取文件大小（字节）
                    file_size = os.path.getsize(file_path)
                    total_size += file_size
                    # 可选：打印每个权重文件的大小，便于核对
                    # print(f"📄 {file}: {round(file_size/(1024*1024), 2)} MB")

        # 4. 转换为MB并四舍五入
        total_size_mb = round(total_size / (1024 * 1024), 2)
        return total_size_mb

    def get_model_size(model_path):
        total_size = 0
        for file in os.listdir(model_path):
            if file.endswith(".bin") or file.endswith(".pt"):
                total_size += os.path.getsize(os.path.join(model_path, file))
        return total_size / (1024 * 1024)  # 转换为MB

    model_size = get_accurate_model_size(model_path)

    # 打印结果
    print("\n======== 轻量化效果验证 ========")
    print(f"模型大小：{model_size:.2f} MB")
    print(f"总推理时间（{sample_num}条样本）：{total_time:.2f} 秒")
    print(f"平均推理时间：{(avg_time_per_sample*1000):.2f} 毫秒")
    print(f"准确率：{accuracy:.4f}")
    print(f"F1值：{f1:.4f}")


if __name__ == '__main__':
    original_model_path = r'C:\Users\gaohu\aiModel\all-MiniLM-L6-v2'
    base_model_path = r'C:\Users\gaohu\aiPyProject\LeetcodeTop\fine_tuning\lora\miniLM\best-model-20260106163605'
    pruned_quant_model_path = r'C:\Users\gaohu\aiPyProject\LeetcodeTop\model_smaller\miniLM\minilm_pruned_quant'
    pruned_distill_quant_model_path = r'C:\Users\gaohu\aiPyProject\LeetcodeTop\model_smaller\miniLM\minilm_pruned_quant_202601061956'
    model_infer_batch(model_path=original_model_path, data_path=VAL_DATA_PATH, sample_num=1000, model_flag="原始模型")
    model_infer_batch(model_path=base_model_path, data_path=VAL_DATA_PATH, sample_num=1000, model_flag="基础微调模型")
    model_infer_batch(model_path=pruned_quant_model_path, data_path=VAL_DATA_PATH, sample_num=1000, model_flag="剪枝量化模型")
    model_infer_batch(model_path=pruned_distill_quant_model_path, data_path=VAL_DATA_PATH, sample_num=1000, model_flag="剪枝蒸馏量化模型")
