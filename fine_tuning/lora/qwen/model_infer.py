import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


if __name__ == '__main__':
    # 1. 加载原模型和分词器
    base_model_path = r"C:\Users\gaohu\aiModel\Qwen2.5-0.5B-Instruct"
    lora_model_path = r"C:\Users\gaohu\aiPyProject\LeetcodeTop\fine_tuning\lora\qwen\output\swift_qwen_finetune"  # 你的LoRA保存目录

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True
    )

    # 2. 挂载LoRA适配器（核心步骤）
    model = PeftModel.from_pretrained(model, lora_model_path)

    # 3. 推理验证（测试简单问答效果）
    prompt = "<|im_start|>user\n苹果的英文是什么？<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.1)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))