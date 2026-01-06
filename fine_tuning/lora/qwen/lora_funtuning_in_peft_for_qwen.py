from peft import LoraConfig
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import TrainingArguments

from transformers import AutoModelForCausalLM, Trainer
from peft import get_peft_model
import swanlab  # 训练监控（可选，替代TensorBoard）

from fine_tuning.data_engineer.alpaca_data_zh.data_format import format_function
# 配置lora参数
lora_config = LoraConfig(
    r=8,  # 低秩维度，适配小模型，避免过拟合
    lora_alpha=16,  # r的2倍，固定配比
    target_modules=["q_proj", "v_proj"],  # Phi-2核心注意力层
    lora_dropout=0.05,  # 防过拟合
    bias="none",
    task_type="CAUSAL_LM"  # 生成类任务
)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="../output/miniLM-finetune-result",  # 训练结果保存路径
    per_device_train_batch_size=1,  # Windows CPU/低配N卡设为1
    gradient_accumulation_steps=4,  # 等效batch_size=4，省显存/CPU资源
    learning_rate=2e-4,  # LoRA专用学习率（Phi-2适配）
    num_train_epochs=3,  # 小模型3轮足够，避免过拟合
    logging_steps=5,  # 每5步打印日志，便于监控
    eval_strategy="epoch",  # 每轮评估验证集
    save_strategy="epoch",  # 每轮保存模型
    load_best_model_at_end=True,  # 训练结束加载最优模型
    fp16=False,  # CPU训练禁用，有N卡可设为True
    gradient_checkpointing=True,  # 省显存/内存（CPU训练也生效）
    weight_decay=0.01,  # 防过拟合
    warmup_ratio=0.1  # 学习率预热，稳训练
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
swanlab.init(project="miniLM-zh-finetune", experiment_name="windows-laptop-test", mode="local")


def print_info(msg: str):
    print(f"{8 * '='}{msg}{8 * '='}")


if __name__ == '__main__':
    model_path = r"C:\Users\gaohu\aiModel\all-MiniLM-L6-v2"  # 改为Qwen/Qwen2.5-0.5B-Instruct【回家下载】
    # 1. 加载Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path, trust_remote_code=True)
    print_info("加载分词器完毕")
    # 2. 加载清洗后的数据
    dataset = load_dataset("json", data_files={
        "train": r"C:\Users\gaohu\aiPyProject\LeetcodeTop\fine_tuning\data\train_data_cleaned.jsonl",
        "validation": r"C:\Users\gaohu\aiPyProject\LeetcodeTop\fine_tuning\data\val_data_cleaned.jsonl"
    })
    print_info("加载训练和验证数据完毕")
    # 3. 预处理数据集（批量处理，提升效率）
    tokenized_dataset = dataset.map(
        format_function,
        batched=True,
        remove_columns=dataset["train"].column_names  # 删除原始列，仅保留编码后的数据
    )
    print_info("预处理数据集完毕")
    # ====================== 加载模型 ======================
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        device_map="auto",  # 自动分配设备（CPU/N卡）
        trust_remote_code=True,
        torch_dtype="auto"  # 自动适配数据类型（CPU用float32，GPU用float16）
    )
    print_info("加载模型完毕")

    # ====================== 注入LoRA适配器 ======================
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # 打印可训练参数（约0.5%，极省资源）

    print_info("注入LoRA适配器完毕")
    # 输出示例：trainable params: 1,048,576 || all params: 2,730,035,200 || trainable%: 0.0384
    # ====================== 定义Trainer ======================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"]
    )
    print_info("定义Trainer完毕")
    # ====================== 启动训练 ======================
    trainer.train()

    # ====================== 保存模型 ======================
    # 仅保存LoRA适配器（体积＜10MB，便于部署）
    model.save_pretrained("./miniLM-lora-adapter")
    tokenizer.save_pretrained("./miniLM-lora-adapter")

    # ====================== 代码核心要点解析 ======================
    # 1. device_map="auto"：Windows下自动识别CPU/GPU，无需手动指定；
    # 2. get_peft_model：仅训练LoRA适配器参数，99.5%的模型参数冻结，省资源；
    # 3. gradient_accumulation_steps=4：CPU训练时，用梯度累加等效增大batch_size，提升训练稳定性；
    # 4. load_best_model_at_end=True：自动选择验证集效果最好的模型，避免过拟合；
    # 5. 仅保存LoRA适配器：无需保存完整模型（2.7B），仅保存适配器（＜10MB），节省磁盘空间。