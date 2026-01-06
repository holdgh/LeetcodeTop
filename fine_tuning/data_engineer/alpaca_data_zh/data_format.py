from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=r"C:\Users\gaohu\aiModel\all-MiniLM-L6-v2", trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token  # 生成模型必须设置pad_token all-MiniLM-L6-v2模型分词器已经具备了pad_token


# 格式化函数（指令微调Prompt模板）
def format_function(examples):
    prompts = []
    labels = []
    for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
        # 拼接Prompt（适配Phi-2的输入格式）
        prompt = f"### Instruction:\n{inst}\n### Input:\n{inp}\n### Response:\n"
        prompts.append(prompt)
        # 标签为输出文本（模型仅学习输出部分）
        labels.append(out)

    # 编码输入
    model_inputs = tokenizer(prompts, max_length=512, truncation=True, padding="max_length")
    # 编码标签
    label_inputs = tokenizer(labels, max_length=256, truncation=True, padding="max_length")
    model_inputs["labels"] = label_inputs["input_ids"]

    return model_inputs
