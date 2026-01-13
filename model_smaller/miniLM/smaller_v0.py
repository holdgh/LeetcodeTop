# -*- coding: utf-8 -*-
"""
MiniLM模型量化+剪枝实战脚本
适用场景：中文情感分类（如酒店/商品评论）
核心流程：剪枝（全连接层）→ 蒸馏补偿精度 → 量化（INT8/FP16）→ 验证效果
"""
import torch
import torch.nn as nn
import numpy as np
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    TrainingArguments, Trainer, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import os
import datetime


model_path_time_str = datetime.datetime.now().strftime(format="%Y%m%d%H%M")


class Pruner:
    def __init__(self, model, pruning_ratio=0.5):
        self.model = model
        self.pruning_ratio = pruning_ratio  # 剪枝比例（全连接层剪50%）
        self.mask = {}  # 保存剪枝掩码（恢复推理用）

    def prune_fc_layers(self):
        """剪枝Transformer的Feed Forward全连接层（MiniLM的核心冗余层）"""
        for name, module in self.model.named_modules():
            # 匹配MiniLM的Feed Forward全连接层（特征维度：384→1536→384）
            if isinstance(module, nn.Linear) and module.in_features in [384, 1536] and module.out_features in [1536,
                                                                                                               384]:
                print(f"剪枝层：{name}, 输入维度：{module.in_features}, 输出维度：{module.out_features}")

                # 1. 计算权重绝对值，筛选低贡献参数
                weight = module.weight.data.cpu().numpy()
                weight_abs = np.abs(weight)
                # 计算剪枝阈值（保留top (1-ratio) 的参数）
                threshold = np.percentile(weight_abs, self.pruning_ratio * 100)

                # 2. 生成剪枝掩码（1=保留，0=剪去）
                mask = (weight_abs >= threshold).astype(np.float32)
                self.mask[name] = torch.tensor(mask, device=DEVICE)

                # 3. 执行剪枝（将低贡献参数置0）
                module.weight.data = module.weight.data * self.mask[name]

                # 4. 冻结偏置（偏置参数少，无需剪枝）
                if module.bias is not None:
                    module.bias.requires_grad = False
        print(f"剪枝完成！全连接层剪枝比例：{self.pruning_ratio * 100}%")
        return self.model

    def apply_mask(self):
        """推理时应用剪枝掩码（防止剪枝参数被更新）"""
        for name, module in self.model.named_modules():
            if name in self.mask:
                module.weight.data = module.weight.data * self.mask[name]


def distill_pruned_model(pruned_model, teacher_model, val_dataset, tokenizer):
    """用原始模型（教师）蒸馏剪枝模型（学生），补偿精度损失"""
    # 蒸馏训练参数（轻量训练，仅1~2个epoch）
    training_args = TrainingArguments(
        output_dir=f"./distill_temp_{model_path_time_str}",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=2,
        learning_rate=5e-6,  # 低学习率，避免破坏剪枝结构
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        weight_decay=0.01,
        fp16=False if DEVICE.type == "cpu" else True,
        gradient_accumulation_steps=8,
        no_cuda=True if DEVICE.type == "cpu" else False,
        report_to="none"
    )

    # 蒸馏损失函数（KL散度 + 分类损失）
    class DistillTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # 学生模型输出
            student_outputs = model(**inputs)
            student_logits = student_outputs.logits
            # 教师模型输出（冻结）
            with torch.no_grad():
                teacher_outputs = teacher_model(**inputs)
                teacher_logits = teacher_outputs.logits
            # KL散度（蒸馏温度=2）
            kl_loss = nn.KLDivLoss()(
                nn.functional.log_softmax(student_logits / 2, dim=-1),
                nn.functional.softmax(teacher_logits / 2, dim=-1)
            ) * (2 * 2)
            # 分类损失
            ce_loss = nn.CrossEntropyLoss()(student_logits, inputs["labels"])
            # 总损失：0.7*KL + 0.3*CE
            loss = 0.7 * kl_loss + 0.3 * ce_loss
            return (loss, student_outputs) if return_outputs else loss

    # 加载验证集（用于蒸馏）
    val_df = pd.read_csv(VAL_DATA_PATH, encoding="utf-8")
    val_df = val_df.drop(labels='dataset', axis=1)  # 仅保留文本及类别标签列
    val_df = val_df.dropna()  # 删除缺省值的行
    val_df = val_df[val_df["sentence"].str.len() > 5].reset_index(drop=True)
    from datasets import Dataset
    val_dataset = Dataset.from_pandas(val_df)

    # 数据预处理
    def preprocess(examples):
        texts = [f"判断情感倾向：{text}" for text in examples["sentence"]]
        encoding = tokenizer(
            texts, max_length=128, truncation=True, padding="max_length", return_tensors="pt"
        )
        encoding["labels"] = torch.tensor(examples["label"], dtype=torch.long)
        return encoding

    tokenized_val = val_dataset.map(preprocess, batched=True, remove_columns=val_dataset.column_names)
    tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # 评估指标
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted")
        }

    # 启动蒸馏
    trainer = DistillTrainer(
        model=pruned_model,
        args=training_args,
        train_dataset=tokenized_val,  # 用验证集轻量蒸馏（避免过拟合）
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics
    )
    trainer.train()
    print("蒸馏完成！剪枝模型精度已补偿")
    return pruned_model


def quantize_model_(model, quant_type="int8"):
    """量化模型（优先量化全连接层和注意力层）"""
    if quant_type == "int8":
        # INT8量化（CPU/移动端友好，精度损失<2%）
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        quant_model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_PATH,
            num_labels=2,
            quantization_config=quant_config,
            device_map={"": DEVICE},
            low_cpu_mem_usage=True
        )
    elif quant_type == "fp16":
        # FP16量化（GPU友好，无精度损失）
        quant_model = model.half() if DEVICE.type == "cuda" else model
    else:
        raise ValueError("量化类型仅支持int8/fp16")

    # 保存量化后的模型
    quant_model.save_pretrained(PRUNED_QUANT_MODEL_PATH)
    tokenizer.save_pretrained(PRUNED_QUANT_MODEL_PATH)
    print(f"{quant_type}量化完成！模型保存至：{PRUNED_QUANT_MODEL_PATH}")
    return quant_model


# 替换原有的quantize_model函数
def quantize_model(model, quant_type="fp16"):
    """CPU适配：用FP16量化替代INT8，避免bitsandbytes依赖"""
    if quant_type == "fp16":
        # PyTorch原生FP16量化（CPU/GPU通用）
        model = model.half() if torch.cuda.is_available() else model
        # 手动量化全连接层权重（进一步减小模型）
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.weight.data = module.weight.data.half()
                if module.bias is not None:
                    module.bias.data = module.bias.data.half()
    else:
        raise ValueError("Windows CPU仅支持fp16量化")

    # 保存量化后的模型
    model.save_pretrained(PRUNED_QUANT_MODEL_PATH, safe_serialization=False)  # 兼容CPU的FP16模型保存
    tokenizer.save_pretrained(PRUNED_QUANT_MODEL_PATH)
    print(f"FP16量化完成！模型保存至：{PRUNED_QUANT_MODEL_PATH}")
    return model


def evaluate_model(model, tokenizer, data_path):
    """验证轻量化后模型的精度和推理速度"""
    # 加载测试数据
    df = pd.read_csv(data_path, encoding="utf-8")
    df = df.drop(labels='dataset', axis=1)  # 仅保留文本及类别标签列
    df = df.dropna()  # 删除缺省值的行
    df = df[df["sentence"].str.len() > 5].reset_index(drop=True)
    texts = df["sentence"].tolist()[:100]  # 取100条样本测试
    labels = df["label"].tolist()[:100]

    # 推理速度测试
    start_time = torch.cuda.Event(enable_timing=True) if DEVICE.type == "cuda" else torch.cuda.Event(
        enable_timing=False)
    end_time = torch.cuda.Event(enable_timing=True) if DEVICE.type == "cuda" else torch.cuda.Event(
        enable_timing=False)

    if DEVICE.type == "cuda":
        start_time.record()
    preds = []
    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                f"判断情感倾向：{text}",
                max_length=128,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(DEVICE)
            output = model(**inputs)
            pred = torch.argmax(output.logits, dim=-1).cpu().item()
            preds.append(pred)
    if DEVICE.type == "cuda":
        end_time.record()
        torch.cuda.synchronize()
        infer_time = start_time.elapsed_time(end_time) / 1000  # 总时间（秒）
    else:
        import time
        infer_time = time.time() - start_time  # CPU计时

    # 精度计算
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")

    # 模型大小计算
    def get_model_size(model_path):
        total_size = 0
        for file in os.listdir(model_path):
            if file.endswith(".bin") or file.endswith(".pt"):
                total_size += os.path.getsize(os.path.join(model_path, file))
        return total_size / (1024 * 1024)  # 转换为MB

    original_size = get_model_size(BASE_MODEL_PATH)
    pruned_quant_size = get_model_size(PRUNED_QUANT_MODEL_PATH)

    # 打印结果
    print("\n======== 轻量化效果验证 ========")
    print(f"原始模型大小：{original_size:.2f} MB")
    print(f"剪枝+量化后模型大小：{pruned_quant_size:.2f} MB")
    print(f"模型压缩比：{original_size / pruned_quant_size:.2f}x")
    print(f"推理速度（100条样本）：{infer_time:.2f} 秒")
    print(f"准确率：{accuracy:.4f}")
    print(f"F1值：{f1:.4f}")


if __name__ == '__main__':
    # ====================== 1. 基础配置（适配CPU/GPU） ======================
    DEVICE = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    # 训练好的MiniLM情感分类模型路径（替换为你的模型路径）
    BASE_MODEL_PATH = r"../../fine_tuning/lora/miniLM/best-model-20260106163605"
    # 轻量化后模型保存路径
    PRUNED_QUANT_MODEL_PATH = f"./minilm_pruned_quant_{model_path_time_str}"
    # 数据集路径（验证用）
    VAL_DATA_PATH = "../../fine_tuning/jd_comment/cleaned_dataset.csv"

    # ====================== 2. 加载原始模型+Tokenizer ======================
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    # 加载原始微调后的模型
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_PATH,
        num_labels=2,
        device_map={"": DEVICE},
        low_cpu_mem_usage=True
    )
    model.eval()

    # ====================== 3. 第一步：剪枝（重点剪全连接层） ======================

    # # 执行剪枝
    # pruner = Pruner(model, pruning_ratio=0.5)
    # pruned_model = pruner.prune_fc_layers()
    # # 应用剪枝掩码（确保推理时剪枝参数不参与计算）
    # pruner.apply_mask()

    # ====================== 4. 第二步：蒸馏补偿（挽回剪枝精度损失） ======================

    # 执行蒸馏（教师模型=原始模型，学生模型=剪枝模型）
    # distilled_pruned_model = distill_pruned_model(pruned_model, model, VAL_DATA_PATH, tokenizer)

    # ====================== 5. 第三步：量化（INT8/FP16） ======================

    # 执行量化
    # quant_type = "int8" if DEVICE.type == "cpu" else "fp16"
    quant_type = "fp16" if DEVICE.type == "cpu" else "int8"
    # final_model = quantize_model(distilled_pruned_model, quant_type=quant_type)
    final_model = quantize_model(model, quant_type=quant_type)
    # final_model = quantize_model(pruned_model, quant_type=quant_type)

    # ====================== 6. 验证轻量化效果 ======================

    # 执行验证
    evaluate_model(final_model, tokenizer, VAL_DATA_PATH)
