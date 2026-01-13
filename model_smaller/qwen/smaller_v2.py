# -*- coding: utf-8 -*-
"""
千问2.5-0.5B 结构化剪枝+蒸馏 标准流程
核心：掩码式剪枝（保维度）+ 蒸馏补偿，适配Windows CPU
"""
import torch
import json
import warnings

warnings.filterwarnings("ignore")

# ====================== 1. 基础依赖导入 ======================
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import torch.nn.utils.prune as prune

# ====================== 2. 基础配置 ======================
# 模型/数据路径
MODEL_PATH = r"C:\Users\gaohu\aiModel\Qwen2.5-0.5B-Instruct"
DATA_PATH = r"C:\Users\gaohu\aiPyProject\LeetcodeTop\fine_tuning\alpaca_zh\alpaca_data_zh_51k.json"
SAVE_PATH = "./qwen2.5-0.5B-pruned-distilled"

# 剪枝配置（结构化，保维度）
PRUNE_RATIO = 0.2  # 剪枝20%冗余权重（仅置零，不删维度）
PRUNE_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]  # 仅剪注意力层

# 训练配置（CPU专用）
MAX_SEQ_LEN = 256
BATCH_SIZE = 1
EPOCHS = 1
LEARNING_RATE = 1e-4

if __name__ == '__main__':
    # ====================== 3. 加载Tokenizer和原始模型（教师模型） ======================
    # 加载Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # 加载教师模型（原始模型，冻结）
    teacher_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
        use_cache=False
    )
    teacher_model.eval()  # 冻结教师模型，仅输出logits


    # ====================== 4. 结构化剪枝（保维度）- 核心步骤 ======================
    def structured_prune_model(model, prune_ratio=0.2, prune_modules=None):
        """
        结构化剪枝：对指定层做L1非结构化剪枝（掩码式，维度不变）
        - prune_ratio：剪枝比例（0~1），仅将低L1权重置零
        - prune_modules：需要剪枝的层名称关键词
        """
        if prune_modules is None:
            prune_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        # 遍历模型，对指定层做剪枝
        for name, module in model.named_modules():
            # 仅对注意力层的线性层剪枝
            if any(kw in name for kw in prune_modules) and isinstance(module, torch.nn.Linear):
                # 对权重做L1剪枝（掩码式，维度不变）
                prune.l1_unstructured(
                    module,
                    name="weight",
                    amount=prune_ratio
                )
                # 移除剪枝掩码（可选，剪枝后固化掩码）
                prune.remove(module, "weight")
                print(f"✅ 完成剪枝：{name}（剪枝比例：{prune_ratio}）")

        print("\n📌 剪枝完成！模型维度未改变，可直接进行蒸馏")
        return model


    # 加载学生模型（待剪枝）
    student_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
        use_cache=False
    )

    # 执行结构化剪枝（核心：保维度）
    student_model = structured_prune_model(student_model, PRUNE_RATIO, PRUNE_MODULES)


    # ====================== 5. 加载并预处理蒸馏数据 ======================
    def load_distill_data():
        """加载少量蒸馏数据（CPU适配，仅用100条）"""
        # 加载数据
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            # raw_data = json.load(f)[:100]
            raw_data = json.load(f)[:10]

        # 格式化千问prompt
        formatted_data = []
        for ex in raw_data:
            instr = ex.get("instruction", "").strip()
            input_txt = ex.get("input", "").strip()
            output_txt = ex.get("output", "").strip()
            if not instr or not output_txt:
                continue

            # 千问标准指令格式
            prompt = f"<|im_start|>user\n{instr}\n{input_txt}<|im_end|>\n<|im_start|>assistant\n{output_txt}<|im_end|>"
            formatted_data.append({"text": prompt})

        # 编码数据
        dataset = Dataset.from_list(formatted_data)

        def encode_fn(examples):
            return tokenizer(
                examples["text"],
                max_length=MAX_SEQ_LEN,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )

        encoded_ds = dataset.map(encode_fn, batched=True)
        encoded_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
        return encoded_ds


    distill_dataset = load_distill_data()
    print(f"\n✅ 蒸馏数据加载完成，共{len(distill_dataset)}条")


    # ====================== 6. 蒸馏训练（补偿剪枝损失） ======================
    # 自定义蒸馏Trainer（计算教师-学生logits MSE损失）
    class DistillTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # 1. 教师模型输出（冻结，无梯度）
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    return_dict=True
                )
                teacher_logits = teacher_outputs.logits

            # 2. 学生模型输出（剪枝后，有梯度）
            student_outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                return_dict=True
            )
            student_logits = student_outputs.logits

            # 3. 蒸馏损失：MSE（仅计算有效序列，避免padding）
            loss_fct = torch.nn.MSELoss()
            # 对齐维度：仅计算[:-1]和[1:]的损失（避免位移错误）
            loss = loss_fct(student_logits[:, :-1, :], teacher_logits[:, 1:, :])

            return (loss, {"student_logits": student_logits}) if return_outputs else loss


    # 训练参数（CPU专用）
    training_args = TrainingArguments(
        output_dir="./qwen-distill-temp",
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        logging_steps=5,
        save_strategy="no",
        fp16=False,  # CPU不支持FP16
        bf16=False,
        use_cpu=True,  # 强制CPU
        report_to="none",  # 关闭日志工具
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
    )

    # 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=student_model,
        padding="max_length",
        max_length=MAX_SEQ_LEN
    )

    # 启动蒸馏
    print("\n🚀 开始蒸馏训练（CPU适配）...")
    trainer = DistillTrainer(
        model=student_model,
        args=training_args,
        train_dataset=distill_dataset,
        data_collator=data_collator
    )
    trainer.train()

    # ====================== 7. 保存剪枝+蒸馏后的模型 ======================
    student_model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print(f"\n🎉 剪枝+蒸馏全部完成！模型保存至：{SAVE_PATH}")
    print(f"📌 模型特性：")
    print(f"  - 剪枝比例：{PRUNE_RATIO * 100}%（注意力层）")
    print(f"  - 维度：与原始模型一致，无维度错误")
    print(f"  - 设备：适配Windows CPU，可直接用于LoRA微调")


    # ====================== 8. 验证模型有效性 ======================
    def validate_model():
        """验证剪枝+蒸馏后模型的生成效果"""
        print("\n🔍 验证模型生成效果：")
        # 加载保存的模型
        valid_model = AutoModelForCausalLM.from_pretrained(
            SAVE_PATH,
            trust_remote_code=True,
            device_map="cpu",
            low_cpu_mem_usage=True
        )

        # 测试生成
        prompt = "<|im_start|>user\n计算1+2×3的结果<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = valid_model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=False
            )

        # 输出结果
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"输入：计算1+2×3的结果")
        print(f"输出：{result.strip()}")


    validate_model()
