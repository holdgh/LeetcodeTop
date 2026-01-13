import datetime
import json

model_path_time_str = datetime.datetime.now().strftime(format="%Y%m%d%H%M%S")
"""
LoRA微调Qwen2.5-0.5B-instruct（适配alpaca_data_zh+Windows CPU）
核心修复：指令格式拼接、维度对齐、CPU环境兼容
版本说明：采用peft框架，结合transformers，自定义数据加载及预处理功能，进行模型微调
结果：流程畅通，小数据量微调loss曲线尚可
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


# ====================== 基础配置（适配你的场景） ======================
# 模型路径（本地/ Hugging Face）
MODEL_PATH = r"C:\Users\gaohu\aiModel\Qwen2.5-0.5B-Instruct"
# alpaca中文数据集路径（替换为你的文件路径）
ALPACA_DATA_PATH = r"C:\Users\gaohu\aiPyProject\LeetcodeTop\fine_tuning\alpaca_zh\alpaca_data_zh_51k.json"
# 训练参数（CPU环境轻量化）
OUTPUT_DIR = f"../output/qwen2.5_instruct_lora_finetune_{model_path_time_str}"
PER_DEVICE_TRAIN_BATCH_SIZE = 1  # CPU必须设为1，避免维度爆炸
GRADIENT_ACCUMULATION_STEPS = 1  # 禁用梯度累积，杜绝维度翻倍
MAX_SEQ_LEN = 512  # 匹配报错中的512维度
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 1  # CPU训练慢，先跑1轮验证
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
    data = data[:100]

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

    print_info("加载并预处理alpaca_data_zh数据集")
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
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
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
    print_info("训练配置")

    swanlab.init(project="qwen25-zh-finetune", experiment_name="windows-laptop-test", mode="local")
    # ====================== 7. 启动训练 ======================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    print_info("启动训练")

    # 最终确认模型配置
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id  # 对齐模型和tokenizer的pad_token_id

    # 启动训练（修复维度不匹配问题）
    print("✅ 开始LoRA微调Qwen2.5-0.5B-Instruct（CPU模式）")
    trainer.train()

    # 保存模型
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ 微调完成！模型保存至：{OUTPUT_DIR}")
