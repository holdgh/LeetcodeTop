# -*- coding: utf-8 -*-
"""
千问2.5-0.5B剪枝+蒸馏+量化（Windows CPU版）
核心：轻量化压缩，适配后续LoRA微调
"""
import json

import torch
from datasets import Dataset
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

QUANT_MODEL_PATH = "./qwen2.5-0.5B-quant"


def prune_qwen_model(model, prune_ratio=0.3):
    """
    剪枝千问模型的注意力层（q_proj/k_proj/v_proj）
    prune_ratio=0.3：删除30%低贡献权重
    """
    for name, param in model.named_parameters():
        if "q_proj" in name or "k_proj" in name or "v_proj" in name:
            # 计算权重的L2范数，保留高贡献权重
            l2_norm = torch.norm(param, dim=-1)
            threshold = torch.quantile(l2_norm, prune_ratio)
            mask = l2_norm > threshold
            param.data = param.data[mask]
    return model


def distill_model(teacher_model, student_model, dataset, tokenizer, max_seq_len=256):
    """
        蒸馏：用原始千问（teacher）指导剪枝模型（student）
        简化版蒸馏，适配CPU
        """
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()  # 回归损失，适配生成任务

    # 仅用100条数据蒸馏（CPU快）
    for i, example in enumerate(dataset.select(range(100))):
        # 编码数据
        inputs = tokenizer(
            example["prompt"],
            max_length=max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        # 教师模型输出（冻结）
        with torch.no_grad():
            teacher_logits = teacher_model(**inputs).logits

        # 学生模型输出
        student_logits = student_model(**inputs).logits

        # 蒸馏损失：让学生logits逼近教师logits
        loss = loss_fn(student_logits, teacher_logits)

        # 反向传播（CPU慢，仅跑少量步数）
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"蒸馏步数{i}，损失：{loss.item():.4f}")

    return student_model


def load_distill_data():
    with open(r"C:\Users\gaohu\aiPyProject\LeetcodeTop\fine_tuning\alpaca_zh\alpaca_data_zh_51k.json", "r", encoding="utf-8") as f:
        data = json.load(f)[:100]

    def format_prompt(example):
        return {"prompt": f"用户：{example['instruction']}\n助手：{example['output']}"}

    return Dataset.from_list(data).map(format_prompt)


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
    model.save_pretrained(QUANT_MODEL_PATH, safe_serialization=False)  # 兼容CPU的FP16模型保存
    tokenizer.save_pretrained(QUANT_MODEL_PATH)
    print(f"FP16量化完成！模型保存至：{QUANT_MODEL_PATH}")
    return model


if __name__ == '__main__':
    # ====================== 1. 加载原始千问模型（CPU） ======================
    MODEL_PATH = r"C:\Users\gaohu\aiModel\Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 加载FP32模型（CPU）
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True
    )

    # ====================== 2. 剪枝（删除冗余参数，CPU轻量化） ======================
    # 简单剪枝：冻结非注意力层，仅保留核心层（CPU友好）

    # 剪枝（CPU下prune_ratio设0.3，避免效果暴跌）
    pruned_model = prune_qwen_model(model, prune_ratio=0.3)
    print("剪枝完毕")
    # ====================== 3. 蒸馏（用原始模型指导剪枝模型，保证效果） ======================

    # 加载少量蒸馏数据（alpaca前100条）
    # from datasets import Dataset
    # import json
    #
    # distill_dataset = load_distill_data()
    # 蒸馏剪枝后的模型
    # distilled_model = distill_model(model, pruned_model, distill_dataset, tokenizer)

    # # ====================== 4. 量化（转为INT4，CPU最优） ======================
    # ====================== 2. 量化（转为INT4，CPU最优） ======================
    # 配置INT4量化（CPU友好，速度最快）
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.float32  # CPU用float32计算
    # )

    # 保存剪枝蒸馏后的模型，再加载量化版
    # distilled_model.save_pretrained("./qwen2.5-0.5B-pruned-distilled")
    # quantized_model = AutoModelForCausalLM.from_pretrained(
    #     "./qwen2.5-0.5B-pruned-distilled",
    #     trust_remote_code=True,
    #     quantization_config=bnb_config,
    #     device_map="cpu",
    #     low_cpu_mem_usage=True
    # )
    # quantized_model = quantize_model(model)
    # 保存最终压缩模型
    # quantized_model.save_pretrained("./qwen2.5-0.5B-compressed")
    # tokenizer.save_pretrained("./qwen2.5-0.5B-compressed")
    # print("✅ 剪枝+蒸馏+量化完成！压缩模型保存至：./qwen2.5-0.5B-compressed")
    # ====================== 3. 蒸馏（用原始模型指导剪枝模型，保证效果） ======================

    # 加载少量蒸馏数据（alpaca前100条）
    # from datasets import Dataset
    # import json
    #
    # distill_dataset = load_distill_data()
    # # 蒸馏剪枝后的模型
    # distilled_model = distill_model(model, quantized_model, distill_dataset, tokenizer)
    # distilled_model.save_pretrained("./qwen2.5-0.5B-pruned-distilled")
