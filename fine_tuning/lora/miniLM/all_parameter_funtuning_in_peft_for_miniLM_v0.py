import torch
# ====================== 关键配置：强制设置数据类型 ======================
torch.set_default_dtype(torch.float32)  # CPU训练默认float32
DEVICE = torch.device("cpu")  # 强制CPU（无GPU时）
from transformers import AutoTokenizer
import swanlab  # 训练监控（可选，替代TensorBoard）
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import re

from fine_tuning.data_engineer.jd_comment.data_format import preprocess_function


def clean_text(text):
    # 删特殊符号/乱码，保留中文
    text = re.sub(r"[^\u4e00-\u9fa5\s]", "", text)
    return text.strip()


# 1. 定义评估指标（分类任务核心：准确率+F1值）
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": accuracy, "f1": f1}


# 2. 训练参数（适配Windows CPU/低配N卡）
training_args = TrainingArguments(
    output_dir="../../output/all-MiniLM-L6-v2-finetune",  # 结果保存路径
    per_device_train_batch_size=8,  # CPU可设4~8，GPU可设16~32
    per_device_eval_batch_size=8,
    num_train_epochs=5,  # 小模型5轮足够，避免过拟合
    learning_rate=2e-5,  # BERT类模型通用学习率
    logging_steps=10,  # 每10步打印日志
    eval_strategy="epoch",  # 每轮评估验证集
    save_strategy="epoch",  # 每轮保存模型
    load_best_model_at_end=True,  # 加载最优模型
    fp16=False,  # CPU禁用，GPU可设True
    weight_decay=0.01,  # 防过拟合
    warmup_ratio=0.1,  # 学习率预热
    metric_for_best_model="f1",  # 以F1值选最优模型
    greater_is_better=True,  # F1值越高越好
    # 修复：禁用自动loss计算（由CustomTrainer手动计算）
    remove_unused_columns=False
)

# ====================== 3. 核心修复：自定义Trainer（绑定Loss计算） ======================
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 1. 前向传播获取logits
        outputs = model(**inputs)
        logits = outputs.logits
        # 2. 获取标签并确保设备一致
        # 3. 提取标签（强制long型）
        labels = inputs["labels"].long()  # 双重保障：强制转为long
        # labels = inputs.get("labels").to(logits.device)
        # 3. 手动计算交叉熵Loss（二分类/多分类通用）
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        # 4. 返回loss（可选返回outputs）
        return (loss, outputs) if return_outputs else loss

# ====================== 初始化监控（可选） ======================
# 第一步：登录（替换为你的API Key）
# swanlab.login(api_key="uC2MLnREQWPijSrCONsqJ")  # 日志保存到远程，采用swanlab.init(……, mode="local")无需登录，日志保存在本地
"""
本地模式下，SwanLab 的日志会保存在./swanlog目录下，训练完成后可通过以下命令在本地查看：
# 打开CMD，进入代码目录，执行
swanlab watch
然后浏览器访问http://127.0.0.1:5092即可查看 Loss、训练步数等监控指标；
若执行swanlab watch提示命令不存在，需将 Python 的 Scripts 目录加入 Windows 环境变量（或用python -m swanlab watch替代）。
"""
# 第二步：初始化
swanlab.init(project="miniLM-zh-finetune", experiment_name="windows-laptop-test", mode="local")


def print_info(msg: str):
    print(f"{8 * '='}{msg}{8 * '='}")


if __name__ == '__main__':
    model_path = r"C:\Users\gaohu\aiModel\all-MiniLM-L6-v2"  # 改为Qwen/Qwen2.5-0.5B-Instruct【回家下载】
    # 1. 加载Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path, trust_remote_code=True)
    print_info("加载分词器完毕")
    # 2. 加载示例数据集（替换为你的本地数据集路径）
    # 示例数据集格式：csv文件，包含text（评论文本）、label（0/1）两列
    df = pd.read_csv("../../jd_comment/dev.csv", encoding="utf-8")
    df = df.drop(labels='dataset', axis=1)  # 仅保留文本及类别标签列
    df = df.dropna()  # 删除缺省值的行
    # 2. 数据清洗（轻量去噪，适配分类任务）
    df["sentence"] = df["sentence"].apply(clean_text)
    df = df[df["sentence"].str.len() > 5]  # 保留有效文本（长度＞5）

    # 划分训练集/验证集/测试集（7:2:1）
    train_df, rest_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label"])
    val_df, test_df = train_test_split(rest_df, test_size=1 / 3, random_state=42, stratify=rest_df["label"])

    # 转换为Hugging Face Dataset格式
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
    print_info("加载训练和验证数据完毕")
    # 3. 预处理数据集（批量处理，提升效率）
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names  # 删除原始列，仅保留编码后数据
    )
    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
        dtype={"labels": torch.long}
    )
    # 4. 设置数据集格式（适配PyTorch）
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    print_info("预处理数据集完毕")
    # ====================== 加载模型 ======================
    from transformers import AutoModelForSequenceClassification

    # 加载模型：num_labels=2（二分类），若多分类则修改为对应类别数
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_path,
        num_labels=2,  # 情感分类为二分类，根据你的任务调整
        problem_type="classification",  # 指定任务类型
        device_map="auto"  # 自动分配设备（CPU/GPU）
    )

    # 验证模型加载成功
    print(f"模型可训练参数：{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    # all-MiniLM-L6-v2全参数约80M，训练成本极低
    print_info("加载模型完毕")
    # 输出示例：trainable params: 1,048,576 || all params: 2,730,035,200 || trainable%: 0.0384
    # ====================== 定义Trainer ======================
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics  # 评估指标
    )
    print_info("定义Trainer完毕")
    # ====================== 启动训练 ======================
    # 开始微调（Windows CPU约30~60分钟，GPU约5~10分钟）
    trainer.train()

    # ====================== 保存模型 ======================
    # 保存最优模型
    trainer.save_model("./best-model")
    tokenizer.save_pretrained("./best-model")

    # ====================== 代码核心要点解析 ======================
    # 1. device_map="auto"：Windows下自动识别CPU/GPU，无需手动指定；
    # 2. get_peft_model：仅训练LoRA适配器参数，99.5%的模型参数冻结，省资源；
    # 3. gradient_accumulation_steps=4：CPU训练时，用梯度累加等效增大batch_size，提升训练稳定性；
    # 4. load_best_model_at_end=True：自动选择验证集效果最好的模型，避免过拟合；
    # 5. 仅保存LoRA适配器：无需保存完整模型（2.7B），仅保存适配器（＜10MB），节省磁盘空间。
