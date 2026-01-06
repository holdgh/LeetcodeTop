import torch

# ====================== 关键配置：强制设置数据类型 ======================
torch.set_default_dtype(torch.float32)  # CPU训练默认float32
DEVICE = torch.device("cpu")  # 强制CPU（无GPU时）
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import swanlab  # 训练监控（可选，替代TensorBoard）
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from imblearn.under_sampling import RandomUnderSampler

import re
import datetime

from fine_tuning.data_engineer.jd_comment.data_format import preprocess_function, preprocess_function_enhanced


# 1. 增强数据清洗（保留核心情感词汇，过滤无意义文本）
def clean_text_enhanced(text):
    if pd.isna(text):
        return ""
    # 保留中文、情感相关标点（！、？、！），过滤广告/无关词汇
    text = re.sub(r"[^\u4e00-\u9fa5\s！？。，]", "", str(text))
    # 过滤无情感倾向的无效文本（如仅含“预订”“咨询”等中性词）
    neutral_words = ["预订", "咨询", "电话", "地址", "价格", "时间"]
    if all(word not in text for word in ["好", "差", "棒", "烂", "差", "满意", "失望"]):
        return ""
    # 统一长度（截断过长，填充过短）
    text = text[:100] if len(text) > 100 else text
    return text.strip()


# 2. 类别均衡处理（若正负样本比例＞2:1，做欠采样/过采样）
def balance_data(df):
    # 统计类别分布
    label_count = df["label"].value_counts()
    print("原始类别分布：", label_count.to_dict())
    # 若类别不均衡（如正:负=3:1），欠采样多数类
    if abs(label_count[0] - label_count[1]) / len(df) > 0.2:
        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(df[["sentence"]], df["label"])
        df_balanced = pd.DataFrame({"sentence": X_resampled["sentence"], "label": y_resampled})
        print("均衡后类别分布：", df_balanced["label"].value_counts().to_dict())
        return df_balanced
    return df


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


# 2. 增强评估指标（添加精准率/召回率，全面判断效果）
def compute_metrics_enhanced(eval_pred):
    logits, labels = eval_pred
    logits = logits.detach().cpu().numpy() if torch.is_tensor(logits) else logits
    labels = labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    precision = precision_score(labels, predictions, average="weighted")
    recall = recall_score(labels, predictions, average="weighted")

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


# 2. 训练参数（适配Windows CPU/低配N卡）
# training_args = TrainingArguments(
#     output_dir="../../output/all-MiniLM-L6-v2-finetune",  # 结果保存路径
#     per_device_train_batch_size=8,  # CPU可设4~8，GPU可设16~32
#     per_device_eval_batch_size=8,
#     num_train_epochs=5,  # 小模型5轮足够，避免过拟合
#     learning_rate=2e-5,  # BERT类模型通用学习率
#     logging_steps=10,  # 每10步打印日志
#     eval_strategy="epoch",  # 每轮评估验证集
#     save_strategy="epoch",  # 每轮保存模型
#     load_best_model_at_end=True,  # 加载最优模型
#     fp16=False,  # CPU禁用，GPU可设True
#     weight_decay=0.01,  # 防过拟合
#     warmup_ratio=0.1,  # 学习率预热
#     metric_for_best_model="f1",  # 以F1值选最优模型
#     greater_is_better=True,  # F1值越高越好
#     # 修复：禁用自动loss计算（由CustomTrainer手动计算）
#     remove_unused_columns=False
# )
model_path_time_str = datetime.datetime.now().strftime(format="%Y%m%d%H%M")
training_args = TrainingArguments(
    output_dir=f"../../output/all-MiniLM-L6-v2-finetune_v2_{model_path_time_str}",
    per_device_train_batch_size=4,  # 增大批次，提升稳定性（CPU可承受）
    per_device_eval_batch_size=4,
    # per_device_train_batch_size=2,  # CPU能跑的最大单Batch
    gradient_accumulation_steps=8,  # 等效Batch=2*16=32（当前最优值）
    gradient_checkpointing=True,  # 开启梯度 checkpoint，减少内存占用（CPU友好）
    num_train_epochs=8,  # 增加轮数，但配合早停避免过拟合
    learning_rate=1e-5,  # 降低学习率，减少震荡（原2e-5→1e-5）
    logging_steps=5,
    eval_strategy="steps",  # 按步数评估，更早发现过拟合
    eval_steps=20,  # 每20步评估一次
    save_strategy="steps",
    save_steps=20,
    load_best_model_at_end=True,
    fp16=False,
    weight_decay=0.05,  # 增大权重衰减，抑制过拟合（原0.01→0.05）
    warmup_ratio=0.2,  # 延长预热，稳定训练初期（原0.1→0.2）
    metric_for_best_model="f1",
    greater_is_better=True,
    remove_unused_columns=False,
    report_to="swanlab",
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    max_steps=1000,  # 仅运行2步就停止，以便恢复训练时查看swanlab日志
    # 新增：早停，避免过拟合
    # early_stopping_patience=3,  # 3次评估无提升则停止
    # early_stopping_threshold=0.001,
)

# ====================== 3. 核心修复：自定义Trainer（绑定Loss计算） ======================
# class CustomTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
#         # 1. 前向传播获取logits
#         outputs = model(**inputs)
#         logits = outputs.logits
#         # 2. 获取标签并确保设备一致
#         # 3. 提取标签（强制long型）
#         labels = inputs["labels"].long()  # 双重保障：强制转为long
#         # labels = inputs.get("labels").to(logits.device)
#         # 3. 手动计算交叉熵Loss（二分类/多分类通用）
#         loss_fct = torch.nn.CrossEntropyLoss()
#         loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
#         # 4. 返回loss（可选返回outputs）
#         return (loss, outputs) if return_outputs else loss


# 1. 改用带标签平滑的CrossEntropyLoss，提升泛化
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"].long()

        # 标签平滑，缓解过拟合
        loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        loss = loss_fct(
            logits.view(-1, model.config.num_labels),
            labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss


# ====================== 5. 自定义Trainer（含平滑Loss） ======================
class SmoothLossTrainer(Trainer):
    def __init__(self, *args, smooth_window=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.smooth_window = smooth_window
        self.loss_history = []

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 强制输入数据到CPU
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"].long()
        loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

        # 移动平均Loss，降低震荡
        # self.loss_history.append(loss.item())
        # if len(self.loss_history) >= self.smooth_window:
        #     smooth_loss = sum(self.loss_history[-self.smooth_window:]) / self.smooth_window
        #     loss = torch.tensor(smooth_loss, device=loss.device)

        return (loss, outputs) if return_outputs else loss

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        # 重写优化器/调度器创建逻辑（核心修复：提前创建优化器）
        super().create_optimizer_and_scheduler(num_training_steps)

        # 替换为余弦退火调度器
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=int(num_training_steps * 0.1),  # 预热10%步数
            num_training_steps=num_training_steps,
            num_cycles=0.5  # 余弦周期，越小下降越平缓
        )

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
# 追加已中断的日志id参数【来自../swanlog/run-20260105_194810-xxnve8zrsysrvem00z2gv】，适用于微调中断场景，恢复swanlab日志记录
# 注意resume和id仅适用于mode为cloud场景，local场景不可用
# swanlab.init(project="miniLM-zh-finetune", experiment_name="windows-laptop-test", mode="local", resume=True, id='xxnve8zrsysrvem00z2gv')
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
    df = pd.read_csv("../../jd_comment/cleaned_dataset.csv", encoding="utf-8")
    df = df.drop(labels='dataset', axis=1)  # 仅保留文本及类别标签列
    df = df.dropna()  # 删除缺省值的行
    # 2. 数据清洗（轻量去噪，适配分类任务）
    df["sentence"] = df["sentence"].apply(clean_text_enhanced)
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
        preprocess_function_enhanced,
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
        device_map={"": DEVICE},
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
        # 新增：添加dropout层，降低过拟合
        hidden_dropout_prob=0.2,  # 隐藏层dropout
        attention_probs_dropout_prob=0.2,  # 注意力层dropout
    )
    # 验证模型加载成功
    print(f"模型可训练参数：{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    # all-MiniLM-L6-v2全参数约80M，训练成本极低
    print_info("加载模型完毕")
    # 输出示例：trainable params: 1,048,576 || all params: 2,730,035,200 || trainable%: 0.0384
    # ====================== 定义Trainer ======================
    # trainer = CustomTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=tokenized_dataset["train"],
    #     eval_dataset=tokenized_dataset["validation"],
    #     compute_metrics=compute_metrics  # 评估指标
    # )
    # 替换原评估函数
    trainer = SmoothLossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics_enhanced  # 新增
    )
    # num_train_steps = trainer.state.max_steps
    # # 余弦退火调度：预热10%步数，之后缓慢下降
    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer=trainer.optimizer,
    #     num_warmup_steps=int(num_train_steps * 0.1),  # 仅预热10%步数
    #     num_training_steps=num_train_steps,
    #     num_cycles=0.5  # 余弦周期设为0.5，下降更平缓
    # )
    # trainer.lr_scheduler = scheduler
    print_info("定义Trainer完毕")
    # ====================== 启动训练 ======================
    # 开始微调（Windows CPU约30~60分钟，GPU约5~10分钟）
    trainer.train()

    # ====================== 保存模型 ======================
    time_str = datetime.datetime.now().strftime(format="%Y%m%d%H%M%S")
    save_path = f"./best-model-{time_str}"
    # 保存最优模型
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)

    # ====================== 代码核心要点解析 ======================
    # 1. device_map="auto"：Windows下自动识别CPU/GPU，无需手动指定；
    # 2. get_peft_model：仅训练LoRA适配器参数，99.5%的模型参数冻结，省资源；
    # 3. gradient_accumulation_steps=4：CPU训练时，用梯度累加等效增大batch_size，提升训练稳定性；
    # 4. load_best_model_at_end=True：自动选择验证集效果最好的模型，避免过拟合；
    # 5. 仅保存LoRA适配器：无需保存完整模型（2.7B），仅保存适配器（＜10MB），节省磁盘空间。
