import datetime
import json
import os
import signal
import sys
from typing import Union, Any, Optional

from torch import nn

model_path_time_str = datetime.datetime.now().strftime(format="%Y%m%d%H%M%S")
"""
LoRA微调Qwen2.5-0.5B-instruct（适配alpaca_data_zh+Windows CPU）
核心修复：指令格式拼接、维度对齐、CPU环境兼容
版本说明：相对于v1，增加了微调中断ctrl+c的数据保存机制
结果：有效
"""
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import swanlab  # 训练监控（可选，替代TensorBoard）

# 全局变量：记录训练状态
train_state = {
    "interrupted": False,
    "current_step": 0,
    "current_loss": 0.0,
    "log_path": "./output/logs/trainer_interrupt.log"
}


def signal_handler(signal_num, frame):
    """捕获中断信号（Ctrl+C），标记为优雅中断"""
    print("\n⚠️  检测到中断信号，开始保存训练状态和日志...")
    train_state["interrupted"] = True
    # 立即保存当前状态到日志
    save_interrupt_log()
    # 正常退出（避免强制终止）
    sys.exit(0)


def save_interrupt_log():
    """保存中断时的训练日志（追加模式，避免覆盖）"""
    log_dir = os.path.dirname(train_state["log_path"])
    os.makedirs(log_dir, exist_ok=True)

    # 构建中断日志
    interrupt_log = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "interrupted_step": train_state["current_step"],
        "interrupted_loss": train_state["current_loss"],
        "status": "gracefully_interrupted"
    }

    # 追加写入（核心：用a模式，保留历史日志）
    with open(train_state["log_path"], "a", encoding="utf-8") as f:
        f.write(json.dumps(interrupt_log, ensure_ascii=False) + "\n")
    print(f"✅ 中断日志已保存至：{train_state['log_path']}")


# 注册中断信号处理器（捕获Ctrl+C）
signal.signal(signal.SIGINT, signal_handler)


# 改造Trainer的训练循环，实时更新训练状态
class SafeTrainer(Trainer):
    def training_step(self,
                      model: nn.Module,
                      inputs: dict[str, Union[torch.Tensor, Any]],
                      num_items_in_batch: Optional[torch.Tensor] = None, ):
        """重写训练步骤，实时记录当前步数和loss"""
        # 执行原始训练步骤
        loss = super().training_step(model, inputs, num_items_in_batch)
        # 更新全局训练状态
        train_state["current_step"] = self.state.global_step
        train_state["current_loss"] = loss.item()
        # 若检测到中断，立即保存模型检查点
        if train_state["interrupted"]:
            # self._save_checkpoint("./output/model/interrupt_checkpoint")
            self._save_checkpoint()  # 具备默认文件名路径 TODO 若存在续训需求，则需要保存中断时的检查点
        return loss


# ====================== 基础配置（适配你的场景） ======================
# 模型路径（本地/ Hugging Face）
MODEL_PATH = r"C:\Users\gaohu\aiModel\Qwen2.5-0.5B-Instruct"
# alpaca中文数据集路径（替换为你的文件路径）
ALPACA_DATA_PATH = r"C:\Users\gaohu\aiPyProject\LeetcodeTop\fine_tuning\alpaca_zh\alpaca_data_zh_51k.json"
# 训练参数（CPU环境轻量化）
OUTPUT_DIR = f"../output/qwen2.5_instruct_lora_finetune_{model_path_time_str}"
PER_DEVICE_TRAIN_BATCH_SIZE = 1  # CPU必须设为1，避免维度爆炸
PER_DEVICE_EVAL_BATCH_SIZE = 2  # 验证批次大小
GRADIENT_ACCUMULATION_STEPS = 1  # 禁用梯度累积，杜绝维度翻倍
MAX_SEQ_LEN = 512  # 匹配报错中的512维度
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 3  # CPU训练慢，先跑1轮验证
LOGGING_STEPS = 5
SAVE_STEPS = 50
# 加载tokenizer，全局可用
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    padding_side="right",  # Qwen2.5必须右padding，否则维度错位
    eos_token="</s>",
    bos_token="<s>",
    pad_token="<pad>"
)


def load_alpaca_data(data_path):
    """加载alpaca中文数据集，按Qwen2.5-Instruct格式拼接"""
    # 加载数据（仅取前100条测试，CPU训练快）
    df = pd.read_csv(data_path, encoding="utf-8").head(100)
    # 填充空值
    df["input"] = df["input"].fillna("")
    df["output"] = df["output"].fillna("")

    # 按Qwen2.5-Instruct的指令格式拼接
    def format_prompt(row):
        if row["input"]:
            prompt = f"""<|im_start|>system
你是一个有用的助手。
<|im_start|>user
{row["instruction"]}
{row["input"]}
<|im_start|>assistant
{row["output"]}<|im_end|>"""
        else:
            prompt = f"""<|im_start|>system
你是一个有用的助手。
<|im_start|>user
{row["instruction"]}
<|im_start|>assistant
{row["output"]}<|im_end|>"""
        return prompt

    df["prompt"] = df.apply(format_prompt, axis=1)
    return Dataset.from_pandas(df[["prompt"]])


def load_alpaca_json_data(data_path):
    """
    加载JSON格式的alpaca中文数据集
    适配两种常见格式：
    1. 单行JSON数组（[{}, {}, ...]）
    2. 每行一个JSON对象（JSON Lines格式）
    """
    # 尝试加载单行JSON数组格式
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"✅ 加载JSON数组格式数据，共{len(data)}条")
    # 加载失败则尝试JSON Lines格式（每行一个JSON）
    except json.JSONDecodeError:
        data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        print(f"✅ 加载JSON Lines格式数据，共{len(data)}条")

    # 仅取前100条快速验证（CPU训练快）TODO 采用1000条数据微调
    data = data[:1000]

    # 转换为Dataset格式，并格式化指令
    def format_prompt(example):
        """按Qwen2.5-Instruct格式拼接指令"""
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output_text = example.get("output", "")

        if input_text:
            prompt = f"""<|im_start|>system
你是一个有用的中文助手。
<|im_start|>user
{instruction}
{input_text}
<|im_start|>assistant
{output_text}<|im_end|>"""
        else:
            prompt = f"""<|im_start|>system
你是一个有用的中文助手。
<|im_start|>user
{instruction}
<|im_start|>assistant
{output_text}<|im_end|>"""
        return {"prompt": prompt}

    # 转换为Hugging Face Dataset并格式化
    dataset = Dataset.from_list(data)
    dataset = dataset.map(format_prompt)
    return dataset


def preprocess_function(examples):
    """预处理函数：核心修复维度对齐问题"""
    # 编码prompt，返回list而非tensor（避免CPU维度混乱）
    model_inputs = tokenizer(
        examples["prompt"],
        max_length=MAX_SEQ_LEN,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
        return_tensors=None  # 关键：返回list，由collator统一处理维度
    )

    # 构建labels：与input_ids一致，pad_token设为-100（损失计算忽略）
    """
    labels代码逻辑拆解：
        第一步：labels和input_ids完全一致
        因果语言模型的训练目标是“根据前文预测下一个token”，因此labels需要和input_ids（输入的token序列）完全对齐——模型输入input_ids的第i个token，需要预测第i+1个token，而labels就是这个“待预测的目标序列”。
        
        第二步：pad_token_id替换为-100
        torch.nn.functional.cross_entropy会忽略label=-100的位置，这样做的目的：
        
        避免计算padding部分的损失（padding是无意义的token，计算损失会干扰训练）；
        保证损失值仅反映“有效文本部分”的生成效果。
    """
    labels = []
    for input_id in model_inputs["input_ids"]:
        label = [token if token != tokenizer.pad_token_id else -100 for token in input_id]
        labels.append(label)

    model_inputs["labels"] = labels
    return model_inputs


def print_info(msg: str):
    print(f"{8 * '='}{msg}完毕{8 * '='}")


if __name__ == '__main__':
    # ====================== 1. 加载Tokenizer（适配Qwen2.5-Instruct） ======================
    # 强制设置pad_token（Qwen2.5-Instruct默认无pad_token，核心修复点）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # 二次确认，避免被覆盖
    print_info("加载Tokenizer")
    # ====================== 2. 加载并预处理alpaca_data_zh数据集 ======================

    # 加载并预处理数据
    dataset = load_alpaca_json_data(ALPACA_DATA_PATH)
    # 批量预处理（禁用多进程，Windows CPU兼容）
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=1
    )
    # 转换为PyTorch数据集（确保维度正确）
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    # 划分训练集和验证集
    split_dataset = tokenized_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    print_info(f"加载并预处理alpaca_data_zh数据集，其中训练集{len(train_dataset)}条，验证集{len(eval_dataset)}条")
    # ====================== 3. 加载Qwen2.5-Instruct模型（CPU适配） ======================
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # CPU必须用float32，避免类型转换维度错
        device_map="cpu",  # 强制CPU，避免自动分配导致维度问题
        low_cpu_mem_usage=True,
        use_cache=False  # 训练时禁用cache，核心修复梯度/维度问题
    )
    # 准备模型用于LoRA训练
    # model = prepare_model_for_training(model)
    # use_gradient_checkpointing=False：CPU下禁用梯度检查点，解决维度/梯度警告
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=False,  # 关键：CPU禁用，避免批次维度错位
        gradient_checkpointing_kwargs=None
    )
    print_info("加载Qwen2.5-Instruct模型")

    # ====================== 4. 配置LoRA（适配Qwen2.5-Instruct） ======================
    lora_config = LoraConfig(
        r=4,  # CPU环境减小r值，降低计算量
        lora_alpha=16,
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        target_modules=["q_proj", "v_proj"],
        # Qwen2.5-Instruct全量LoRA层
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False
    )
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    # 打印可训练参数（验证配置）
    model.print_trainable_parameters()
    print_info("配置LoRA")

    # ====================== 5. 数据整理器（核心修复维度对齐） ======================
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        return_tensors="pt",
        padding="max_length",
        max_length=MAX_SEQ_LEN,
        label_pad_token_id=-100  # 关键：labels的pad_token设为-100
    )
    print_info("数据整理器")

    # ====================== 6. 训练配置（CPU专用） ======================
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,  # 平均每步评估有多少条验证数据参与
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        logging_steps=LOGGING_STEPS,
        # save_steps=SAVE_STEPS,
        save_strategy="no",  # 不保存中间模型
        eval_strategy="steps",  # 按步数评估，更早发现过拟合
        eval_steps=10,  # 每10步评估一次
        fp16=False,  # CPU禁用FP16
        bf16=False,
        weight_decay=0.01,
        warmup_steps=10,
        logging_dir="./logs",
        report_to="swanlab",  # 禁用wandb
        remove_unused_columns=False,  # 保留labels列
        load_best_model_at_end=False,
        # 核心修复：禁用梯度检查点，避免CPU梯度警告
        gradient_checkpointing=False,
        # 确保损失计算维度匹配
        label_smoothing_factor=0.0,
        max_grad_norm=1.0
    )
    """
    GPU环境下的训练配置参数：
        training_args = TrainingArguments(
            # 基础输出配置
            output_dir=OUTPUT_DIR,
            overwrite_output_dir=True,  # 覆盖已有输出目录，适合多次调试
            
            # ===================== 核心优化：利用GPU提升batch_size，降低Loss波动 =====================
            per_device_train_batch_size=8,  # GPU可支撑更大batch（原CPU=1），大幅降低梯度噪声
            per_device_eval_batch_size=4,   # 验证集batch同步提升，评估更稳定
            gradient_accumulation_steps=2,  # 等效batch_size=8*2=16，进一步平滑梯度
            max_grad_norm=1.0,              # 梯度裁剪，防止大batch下梯度爆炸
            
            # ===================== 学习率与预热优化：适配大batch，避免过拟合 =====================
            learning_rate=1e-4,             # 原2e-4偏大，GPU大batch下调低至1e-4，提升收敛稳定性
            warmup_steps=100,               # 延长预热步数（原10），适配大batch的梯度更新节奏
            weight_decay=0.05,              # 提高权重衰减（原0.01），小数据集下增强正则化，避免过拟合
            
            # ===================== 训练周期与评估策略：平衡效率与效果 =====================
            num_train_epochs=5,             # 适当增加epoch（原3），GPU训练更快，充分利用小数据集
            eval_strategy="steps",          # 保留按步评估，更早发现过拟合
            eval_steps=50,                  # 降低评估频率（原10），减少GPU算力浪费，提升训练效率
            logging_steps=LOGGING_STEPS,    # 保持原日志步长，便于对比Loss变化
            
            # ===================== GPU专属优化：混合精度+梯度检查点 =====================
            fp16=True,                      # 开启FP16混合精度训练，GPU提速30%-50%，降低显存占用
            bf16=False,                     # 若使用A100等支持BF16的GPU可开启，否则保持False
            gradient_checkpointing=True,    # 开启梯度检查点，显存占用降低30%+，支持更大batch/模型
            
            # ===================== 模型保存与早停：避免过拟合，留存最优模型 =====================
            save_strategy="steps",          # 按步保存模型（原no），留存训练过程中的模型
            save_total_limit=3,             # 仅保留最近3个模型，避免磁盘占满
            load_best_model_at_end=True,    # 训练结束后加载验证Loss最低的模型，避免过拟合 【要求保存策略和验证策略保持一致，也即eval_strategy="steps"时，要求save_strategy="steps"】
            metric_for_best_model="eval_loss",  # 以验证Loss为最优模型判定标准
            greater_is_better=False,        # eval_loss越小越好
            
            # ===================== 其他关键配置：适配中文微调 =====================
            logging_dir="./logs/gpu",       # 单独的GPU训练日志目录，便于区分
            report_to="tensorboard",        # 改用tensorboard可视化（替代swanlab，更通用）
            remove_unused_columns=False,    # 保留labels列，避免数据丢失
            label_smoothing_factor=0.1,     # 轻微标签平滑（原0.0），提升模型泛化能力
            seed=42,                        # 固定随机种子，保证实验可复现
            dataloader_pin_memory=True,     # GPU下开启pin_memory，提升数据加载速度
            dataloader_num_workers=4,       # 多线程加载数据（CPU建议0，GPU建议2-4）
        )
    """
    print_info("训练配置")

    swanlab.init(project="qwen25-zh-finetune", experiment_name="windows-laptop-test", mode="local")
    # ====================== 7. 启动训练 ======================
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=tokenized_dataset,
    #     data_collator=data_collator,
    # )
    # print_info("启动训练")

    # 最终确认模型配置
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id  # 对齐模型和tokenizer的pad_token_id

    # 启动训练（修复维度不匹配问题）
    print("✅ 开始LoRA微调Qwen2.5-0.5B-Instruct（CPU模式）")
    # trainer.train()

    trainer = SafeTrainer(  # 替换为SafeTrainer
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    print_info("启动训练")
    # 训练时捕获中断异常，确保日志保存
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n🚨 训练被手动终止，正在保存最终状态...")
        # 保存最新的模型检查点
        trainer.save_model("./output/model/last_interrupted_model")
        # 保存完整的训练日志
        with open("./output/logs/full_trainer_log.json", "w", encoding="utf-8") as f:
            json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=4)
        print("✅ 所有日志和模型已保存完成！")

    # 保存模型
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ 微调完成！模型保存至：{OUTPUT_DIR}")
