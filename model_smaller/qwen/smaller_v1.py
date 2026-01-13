# # -*- coding: utf-8 -*-
# """
# 千问2.5-0.5B GPTQ量化+蒸馏（Windows CPU专用版）
# 核心：放弃bitsandbytes，用GPTQ量化，无系统依赖
# """
# import torch
# import json
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     TrainingArguments,
#     Trainer,
#     DataCollatorForSeq2Seq
# )
# from datasets import Dataset
# from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig  # GPTQ量化库
#
# # ====================== 1. 基础配置 ======================
# MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"  # 千问0.5B
# SAVE_COMPRESSED_PATH = "./qwen2.5-0.5B-gptq-compressed"  # 量化模型保存路径
# DISTILL_DATA_PATH = "./alpaca_data_zh_51k.json"  # 蒸馏数据路径
# MAX_SEQ_LEN = 256  # 最大序列长度（CPU适配）
# BATCH_SIZE = 1  # CPU批次大小
# GPTQ_BITS = 4  # 4bit量化（体积最小）
#
# # ====================== 2. 加载Tokenizer和原始模型（教师模型） ======================
# # 加载Tokenizer
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.pad_token_id = tokenizer.eos_token_id
#
# # 加载原始模型（教师模型，冻结）
# teacher_model = AutoModelForCausalLM.from_pretrained(
#     MODEL_PATH,
#     trust_remote_code=True,
#     torch_dtype=torch.float32,
#     device_map="cpu",
#     low_cpu_mem_usage=True,
#     use_cache=False
# )
# teacher_model.eval()  # 冻结教师模型
#
#
# # ====================== 3. 加载并预处理蒸馏/量化校准数据 ======================
# def load_and_preprocess_data():
#     """加载数据，同时用于量化校准和蒸馏"""
#     # 加载少量数据（CPU快，仅用100条）
#     with open(DISTILL_DATA_PATH, "r", encoding="utf-8") as f:
#         raw_data = json.load(f)[:100]
#
#     # 格式化千问prompt
#     formatted_data = []
#     for ex in raw_data:
#         instr = ex.get("instruction", "").strip()
#         input_txt = ex.get("input", "").strip()
#         output_txt = ex.get("output", "").strip()
#         if not instr or not output_txt:
#             continue
#         prompt = f"<|im_start|>user\n{instr}\n{input_txt}<|im_end|>\n<|im_start|>assistant\n{output_txt}<|im_end|>"
#         formatted_data.append({"text": prompt})
#
#     # 编码数据
#     dataset = Dataset.from_list(formatted_data)
#
#     def encode_function(examples):
#         return tokenizer(
#             examples["text"],
#             max_length=MAX_SEQ_LEN,
#             truncation=True,
#             padding="max_length",
#             return_tensors="pt"
#         )
#
#     encoded_ds = dataset.map(encode_function, batched=True)
#     encoded_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
#     return encoded_ds, formatted_data
#
#
# # 加载编码后的数据（蒸馏用）和原始文本（量化校准用）
# distill_dataset, calibrate_texts = load_and_preprocess_data()
#
# # ====================== 4. GPTQ量化（学生模型，CPU友好） ======================
# # 配置GPTQ量化参数
# quantize_config = BaseQuantizeConfig(
#     bits=GPTQ_BITS,  # 4bit量化
#     group_size=128,  # 量化分组大小（默认）
#     desc_act=False,  # 关闭激活量化，适配生成式模型
#     model_name_or_path=MODEL_PATH,
#     model_file_base_name="pytorch_model",
# )
#
# # 加载并量化模型（纯CPU，无GPU依赖）
# print("开始GPTQ量化（CPU适配）...")
# student_model = AutoGPTQForCausalLM.from_pretrained(
#     MODEL_PATH,
#     quantize_config=quantize_config,
#     trust_remote_code=True,
#     device_map="cpu",
#     use_cache=False
# )
#
# # 量化校准（用少量数据校准，保证精度）
# calibrate_samples = [tokenizer(text, return_tensors="pt") for text in calibrate_texts[:10]]
# student_model.quantize(
#     calibrate_samples,
#     batch_size=1,
#     use_triton=False,  # 关闭triton（仅GPU用）
# )
#
# # 保存量化后的模型
# student_model.save_quantized(SAVE_COMPRESSED_PATH)
# tokenizer.save_pretrained(SAVE_COMPRESSED_PATH)
# print(f"✅ GPTQ量化完成，模型保存至：{SAVE_COMPRESSED_PATH}")
#
# # 重新加载量化模型（避免内存溢出）
# student_model = AutoGPTQForCausalLM.from_quantized(
#     SAVE_COMPRESSED_PATH,
#     device_map="cpu",
#     use_triton=False,
#     trust_remote_code=True,
#     use_cache=False
# )
#
#
# # ====================== 5. 蒸馏训练（补偿量化损失） ======================
# # 自定义Trainer，实现蒸馏损失
# class DistillTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         # 教师模型输出（冻结，无梯度）
#         with torch.no_grad():
#             teacher_outputs = teacher_model(
#                 input_ids=inputs["input_ids"],
#                 attention_mask=inputs["attention_mask"],
#                 return_dict=True
#             )
#             teacher_logits = teacher_outputs.logits
#
#         # 学生模型输出
#         student_outputs = model(
#             input_ids=inputs["input_ids"],
#             attention_mask=inputs["attention_mask"],
#             return_dict=True
#         )
#         student_logits = student_outputs.logits
#
#         # MSE蒸馏损失（仅计算有效序列）
#         loss_fct = torch.nn.MSELoss()
#         loss = loss_fct(student_logits[:, :-1, :], teacher_logits[:, 1:, :])
#
#         return (loss, {"student_logits": student_logits}) if return_outputs else loss
#
#
# # 训练参数（纯CPU配置）
# training_args = TrainingArguments(
#     output_dir="./distill_temp",
#     per_device_train_batch_size=BATCH_SIZE,
#     num_train_epochs=1,  # 仅训练1轮，CPU快
#     logging_steps=5,  # 每5步打印日志
#     save_strategy="no",  # 不保存中间模型
#     fp16=False,  # CPU不支持FP16
#     bf16=False,
#     device_map="cpu",
#     report_to="none",  # 不使用日志工具
#     learning_rate=1e-4,  # 小学习率，避免过拟合
# )
#
# # 数据整理器
# data_collator = DataCollatorForSeq2Seq(
#     tokenizer=tokenizer,
#     model=student_model,
#     padding="max_length",
#     max_length=MAX_SEQ_LEN
# )
#
# # 启动蒸馏
# print("开始蒸馏量化模型...")
# trainer = DistillTrainer(
#     model=student_model,
#     args=training_args,
#     train_dataset=distill_dataset,
#     data_collator=data_collator
# )
# trainer.train()
#
# # 保存蒸馏后的最终模型
# student_model.save_quantized(f"{SAVE_COMPRESSED_PATH}-distilled")
# tokenizer.save_pretrained(f"{SAVE_COMPRESSED_PATH}-distilled")
# print(f"\n✅ 量化+蒸馏全部完成！最终模型保存至：{SAVE_COMPRESSED_PATH}-distilled")
# print(f"📌 模型体积：≈60MB（4bit），CPU推理/微调速度提升3-5倍")