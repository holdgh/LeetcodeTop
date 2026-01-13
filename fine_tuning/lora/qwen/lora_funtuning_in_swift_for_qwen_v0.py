# -*- coding: utf-8 -*-
"""
Swift框架+Qwen2.5-0.5B 专业领域微调示例
核心：自动适配Qwen的提示词模板+数据预处理
版本说明：采用swift框架，进行数据加载及预处理功能，模型微调。
结果：未跑通逻辑
"""
from swanlab.integration.transformers import SwanLabCallback
from swift.llm import get_model_tokenizer, get_template, load_dataset, EncodePreprocessor
from swift.tuners import Swift, LoraConfig
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainingArguments, Trainer
from swift.utils import seed_everything, get_model_parameter_info, get_logger, find_all_linears
import swanlab  # 训练监控（可选，替代TensorBoard）

logger = get_logger()
if __name__ == '__main__':
    # ====================== 1. 基础配置 ======================
    # 模型路径（支持本地/ HuggingFace Hub）
    MODEL_ID = r"C:\Users\gaohu\aiModel\Qwen2.5-0.5B-Instruct"
    # 数据集路径（你的专业领域数据）
    DATA_PATH = [r"C:\Users\gaohu\aiPyProject\LeetcodeTop\fine_tuning\alpaca_zh\alpaca_data_zh_1h.json"]
    # 输出目录
    OUTPUT_DIR = "./output/swift_qwen_finetune"
    # 随机种子（确保复现）
    seed_everything(42)
    data_seed = 42
    max_length = 2048
    split_dataset_ratio = 0.1  # Split validation set
    num_proc = 4  # The number of processes for data loading.
    system = 'You are a helpful assistant.'
    # ====================== 2. 加载模型、Tokenizer、提示词模板 ======================
    # Swift自动加载Qwen2.5的模型和Tokenizer
    model, tokenizer = get_model_tokenizer(
        MODEL_ID,
        model_kwargs={
            "torch_dtype": "float32",
            "device_map": "cpu",  # 适配Windows CPU
            # "low_cpu_mem_usage": True,
            # "use_cache": False
        }
    )

    # Swift自动加载Qwen2.5的提示词模板（无需手动拼接格式）
    template = get_template(model.model_meta.template, tokenizer, default_system=system, max_length=max_length)
    template.set_mode('train')

    target_modules = find_all_linears(model)
    # template = get_template("qwen")  # Qwen专属模板，自动生成<|im_start|>格式
    # logger.info(f"Qwen提示词模板示例：\n{template.get_example()}")

    # ====================== 3. 配置LoRA（Swift封装，简化参数） ======================
    lora_config = LoraConfig(
        task_type='CAUSAL_LM',
        r=16,  # LoRA秩，专业领域建议16~32
        target_modules=["q_proj", "v_proj"],  # Qwen2.5的注意力层
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none"
    )
    model = Swift.prepare_model(model, lora_config)  # 自动注入LoRA
    logger.info(f'model: {model}')
    model_parameter_info = get_model_parameter_info(model)
    logger.info(f'model_parameter_info: {model_parameter_info}')

    # ====================== 4. 配置数据集（Swift自动预处理） ======================
    # dataset_config = DatasetConfig(
    #     path=DATA_PATH,
    #     type="json",  # 数据格式为json
    #     # Swift自动将question-answer映射为Qwen的prompt格式
    #     instruction_key="question",  # 原始数据中的“问题”字段
    #     output_key="answer",  # 原始数据中的“答案”字段
    #     max_length=512,  # 专业领域建议512
    #     train_ratio=0.8  # 划分80%为训练集
    # )
    train_dataset, val_dataset = load_dataset(DATA_PATH, split_dataset_ratio=split_dataset_ratio, num_proc=num_proc,
                                              seed=data_seed)  # alpaca_data_zh数据格式：每一条数据包含3个字段，instruction【用户输入】, input【用户输入的补充说明】, output【模型输出】

    logger.info(f'train_dataset: {train_dataset}')
    logger.info(f'val_dataset: {val_dataset}')
    logger.info(f'train_dataset[0]: {train_dataset[0]}')

    train_dataset = EncodePreprocessor(template=template)(train_dataset, num_proc=num_proc)
    val_dataset = EncodePreprocessor(template=template)(val_dataset, num_proc=num_proc)
    logger.info(f'encoded_train_dataset[0]: {train_dataset[0]}')

    # Print a sample
    template.print_inputs(train_dataset[0])
    # ====================== 5. 配置训练参数（Swift封装，适配大模型） ======================
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,  # CPU模拟大批次
        learning_rate=5e-4,  # Qwen2.5 LoRA微调建议5e-4
        num_train_epochs=1,  # 专业领域数据建议3~5轮
        logging_steps=1,
        save_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=False,  # CPU关闭FP16
        bf16=False,
        load_best_model_at_end=True,
        # report_to=["swanlab"]  # 关闭日志工具
        # CPU关键配置
        dataloader_pin_memory=False,  # 强制关闭pin_memory
        dataloader_num_workers=0,  # Windows下禁用多线程加载
        remove_unused_columns=False,  # 避免数据列丢失
        gradient_checkpointing=False  # CPU下关闭梯度检查点（减少阻塞）
    )

    # swanlab.init(project="qwen25-zh-finetune", experiment_name="windows-laptop-test", mode="local")
    swanlab_callback = SwanLabCallback(
        project="qwen25-zh-finetune",
        experiment_name="lora-qwen2-0.5b",
        mode="local",
        description="Lora微调一个Qwen2-0.5B模型"
    )
    # ====================== 6. 启动SFT微调（Swift封装，一键运行） ======================
    # run_sft_train(
    #     model=model,
    #     tokenizer=tokenizer,
    #     template=template,
    #     dataset_config=dataset_config,
    #     training_args=training_args
    # )
    model.enable_input_require_grads()  # Compatible with gradient checkpointing
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id  # 对齐模型和tokenizer的pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id  # 对齐模型和tokenizer的pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id  # 对齐模型和tokenizer的pad_token_id
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=template.data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        template=template,
        callbacks=[swanlab_callback]
    )
    trainer.train()

    last_model_checkpoint = trainer.state.last_model_checkpoint
    logger.info(f'last_model_checkpoint: {last_model_checkpoint}')

    # ====================== 7. 推理测试（Swift自动加载微调后的模型） ======================
    # from swift import infer
    # # 加载微调后的模型
    # infer_model = Swift.from_pretrained(
    #     model,
    #     os.path.join(OUTPUT_DIR, "best_model"),
    #     lora_config=lora_config
    # )
    #
    # # 专业领域问题测试
    # question = "请解释金融领域中远期合约的信用风险特征"
    # response = infer(
    #     infer_model,
    #     template,
    #     question,
    #     tokenizer=tokenizer,
    #     max_new_tokens=200,
    #     temperature=0.7
    # )
    # logger.info(f"\n问题：{question}")
    # logger.info(f"回答：{response}")
