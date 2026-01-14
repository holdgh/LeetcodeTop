import datetime
import json
import os
import signal
import sys
from typing import Union, Any, Optional

from torch import nn

model_path_time_str = datetime.datetime.now().strftime(format="%Y%m%d%H%M%S")
"""
LoRA微调Qwen2.5-0.5B-instruct（适配alpaca_data_zh+Linux单卡GPU）
核心修复：GPU设备适配、梯度计算逻辑、LoRA参数有效性、混合精度训练
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
# import bitsandbytes as bnb  # GPU量化训练依赖
import numpy as np
import swanlab  # 训练监控

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
    save_interrupt_log()
    sys.exit(0)


def save_interrupt_log():
    """保存中断时的训练日志（追加模式）"""
    log_dir = os.path.dirname(train_state["log_path"])
    os.makedirs(log_dir, exist_ok=True)

    interrupt_log = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "interrupted_step": train_state["current_step"],
        "interrupted_loss": train_state["current_loss"],
        "status": "gracefully_interrupted"
    }

    with open(train_state["log_path"], "a", encoding="utf-8") as f:
        f.write(json.dumps(interrupt_log, ensure_ascii=False) + "\n")
    print(f"✅ 中断日志已保存至：{train_state['log_path']}")


# 注册中断信号处理器
signal.signal(signal.SIGINT, signal_handler)


# 改造Trainer的训练循环，实时更新训练状态
class SafeTrainer(Trainer):
    def training_step(self,
                      model: nn.Module,
                      inputs: dict[str, Union[torch.Tensor, Any]],
                      num_items_in_batch: Optional[torch.Tensor] = None, ):
        """重写训练步骤，实时记录当前步数和loss"""
        loss = super().training_step(model, inputs, num_items_in_batch)
        # 更新全局训练状态
        train_state["current_step"] = self.state.global_step
        train_state["current_loss"] = loss.item()
        # 若检测到中断，立即保存模型检查点
        if train_state["interrupted"]:
            self._save_checkpoint()
        return loss


# ====================== 基础配置（适配Linux单卡GPU） ======================
# 强制指定GPU设备（核心修改1：GPU环境初始化）
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"  # 优化GPU显存分配
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 因为CUDA_VISIBLE_DEVICES=2，所以这里显示cuda:0（映射到物理GPU2）
print(f"当前使用设备：{DEVICE}")

# 模型/数据路径
# MODEL_PATH = "/nas_data/models/Qwen/Qwen2.5-0.5B-Instruct"
MODEL_PATH = "/nas_data/models/Qwen/Qwen2.5-7B-Instruct"
ALPACA_DATA_PATH = "../../alpaca_zh/alpaca_data_zh_51k.json"
# 训练参数（GPU适配，降低批次避免OOM）
OUTPUT_DIR = f"../output/qwen2.5_instruct_lora_finetune_{model_path_time_str}"
PER_DEVICE_TRAIN_BATCH_SIZE = 2  # 8G GPU推荐2，16G GPU可设4/8
PER_DEVICE_EVAL_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 16  # 梯度累积模拟大批次
MAX_SEQ_LEN = 512
LEARNING_RATE = 1e-4
NUM_TRAIN_EPOCHS = 5
LOGGING_STEPS = 10
SAVE_STEPS = 1000
DATA_NUM = 30000

# 加载tokenizer（核心修改2：对齐pad_token）
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    padding_side="right",
    eos_token="</s>",
    bos_token="<s>",
    pad_token="<pad>"
)
# 强制对齐pad_token_id（避免模型/Tokenizer不匹配）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id


def load_alpaca_json_data(data_path):
    """加载JSON格式的alpaca中文数据集"""
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"✅ 加载JSON数组格式数据，共{len(data)}条")
    except json.JSONDecodeError:
        data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        print(f"✅ 加载JSON Lines格式数据，共{len(data)}条")

    # GPU训练可适当增加数据量（这里取3000条平衡速度和效果）
    data = data[:DATA_NUM]

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

    dataset = Dataset.from_list(data)
    dataset = dataset.map(format_prompt)
    return dataset


# def preprocess_function(examples):
#     """预处理函数：GPU适配"""
#     model_inputs = tokenizer(
#         examples["prompt"],
#         max_length=MAX_SEQ_LEN,
#         truncation=True,
#         padding="max_length",
#         return_attention_mask=True,
#         return_tensors=None
#     )

#     # 构建labels：pad_token设为-100
#     labels = []
#     for input_id in model_inputs["input_ids"]:
#         label = [token if token != tokenizer.pad_token_id else -100 for token in input_id]
#         labels.append(label)

#     model_inputs["labels"] = labels
#     return model_inputs


def preprocess_function(examples):
    """预处理函数：核心修复数据张量格式，解决无梯度问题"""
    model_inputs = tokenizer(
        examples["prompt"],
        max_length=MAX_SEQ_LEN,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
        return_tensors=None
    )

    # 核心修复：将labels转为numpy数组再转list，避免list of numpy（解决张量警告）
    labels = []
    for input_id in model_inputs["input_ids"]:
        # 替换pad_token_id为-100
        label = np.array([token if token != tokenizer.pad_token_id else -100 for token in input_id])
        labels.append(label.tolist())  # 转为list，保持格式兼容

    # 强制转为numpy数组后再赋值（解决张量创建慢的警告）
    model_inputs["input_ids"] = np.array(model_inputs["input_ids"])
    model_inputs["attention_mask"] = np.array(model_inputs["attention_mask"])
    model_inputs["labels"] = np.array(labels)

    return model_inputs


def print_info(msg: str):
    print(f"{8 * '='}{msg}{8 * '='}")


if __name__ == '__main__':
    print_info("加载Tokenizer")

    # ====================== 加载并预处理数据 ======================
    dataset = load_alpaca_json_data(ALPACA_DATA_PATH)
    # 批量预处理（Linux GPU可启用多进程）
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=1  # Linux下可用多进程加速
    )
    # 转换为PyTorch数据集
    # tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
        device=DEVICE  # 强制数据张量在GPU
    )
    # 划分训练集和验证集
    split_dataset = tokenized_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    print_info(f"加载并预处理数据集，训练集{len(train_dataset)}条，验证集{len(eval_dataset)}条")

    # ====================== 加载模型（核心修改3：GPU适配） ======================
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16,  # GPU用float16节省显存
        device_map={"": 0},  # 自动分配到GPU
        low_cpu_mem_usage=True,
        use_cache=False  # 训练时禁用cache
    )
    # 准备模型用于LoRA训练（GPU版）
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=False,  # GPU启用梯度检查点节省显存
    )
    # 强制模型部署到GPU
    model = model.to(DEVICE)
    print_info("加载Qwen2.5-Instruct模型并部署到GPU")

    # ====================== 配置LoRA（核心修改4：确保可训练参数） ======================
    lora_config = LoraConfig(
        r=8,  # GPU可适当增大r值
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # Qwen2.5-0.5B核心注意力层
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False
    )
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    # 打印可训练参数（关键：确认trainable params > 0）
    model.print_trainable_parameters()
    # 强制开启梯度（修复无梯度问题）
    for name, param in model.named_parameters():
        if "lora" in name or "LoRA" in name:
            param.requires_grad = True
            param.data = param.data.to(DEVICE)  # 强制LoRA参数在GPU
    print_info("配置LoRA并确认可训练参数")
    model.train()  # 强制训练模式
    # ====================== 数据整理器 ======================
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        return_tensors="pt",
        padding="max_length",
        max_length=MAX_SEQ_LEN,
        label_pad_token_id=-100,
    )
    print_info("初始化数据整理器")

    # ====================== 训练配置（核心修改5：GPU专属） ======================
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=50,
        fp16=True,  # GPU启用FP16混合精度
        bf16=False,
        weight_decay=0.05,
        warmup_steps=100,
        logging_dir="./logs",
        report_to="swanlab",
        remove_unused_columns=False,
        load_best_model_at_end=True,
        gradient_checkpointing=False,  # GPU启用梯度检查点
        label_smoothing_factor=0.0,
        max_grad_norm=1.0,
        # 核心修复：禁用accelerate的分布式逻辑（单卡GPU）
        disable_tqdm=False,
        dataloader_pin_memory=True,  # GPU启用pin_memory加速
        # dataloader_num_workers=4,
        dataloader_num_workers=0,  # 单卡禁用多进程，避免数据设备冲突
        # 禁用分布式，强制单卡
        no_cuda=False,
        local_rank=-1,
    )
    print_info("初始化训练参数")

    # ====================== SwanLab配置 ======================
    swanlab.login(api_key="uC2MLnREQWPijSrCONsqJ")
    swanlab.init(project="qwen25-zh-finetune", experiment_name=f"linux-gpu-test_{DATA_NUM}_{NUM_TRAIN_EPOCHS}_{LEARNING_RATE}")

    # ====================== 启动训练 ======================
    # 最终确认模型配置
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    trainer = SafeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    print_info("启动GPU版LoRA微调")

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n🚨 训练被手动终止，正在保存最终状态...")
        trainer.save_model("./output/model/last_interrupted_model")
        with open("./output/logs/full_trainer_log.json", "w", encoding="utf-8") as f:
            json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=4)
        print("✅ 日志和模型已保存！")

    # 保存最终模型
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ 微调完成！模型保存至：{OUTPUT_DIR}")
