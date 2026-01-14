import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

if __name__ == '__main__':
    # 1. 加载原模型和分词器
    base_model_path = r"C:\Users\gaohu\aiModel\Qwen2.5-0.5B-Instruct"
    lora_model_path = r"C:\Users\gaohu\aiPyProject\LeetcodeTop\fine_tuning\lora\qwen\qwen2.5_instruct_lora_finetune_20260113112206"  # 你的LoRA保存目录

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
    # prompt = "<|im_start|>user\n写一篇比较和对比两种不同物品的文章。\n足球和篮球<|im_end|>\n<|im_start|>assistant\n"
    prompt = """<<|im_start|>system
你是一个对比分析助手，必须严格按照以下模板回答，不得修改格式，不得捏造事实：
1. 维度1：参赛人数
- 足球：每队11人，全场共22人参赛
- 篮球：每队5人，全场共10人参赛
2. 维度2：场地规格
- 足球：长方形场地，长度90-120米，宽度45-90米，室外草地
- 篮球：长方形场地，长度28米，宽度15米，室内/室外塑胶地
3. 维度3：比赛时长
- 足球：标准比赛90分钟，分上下半场各45分钟，中场休息15分钟
- 篮球：NBA比赛48分钟，分4节各12分钟；FIBA比赛40分钟，分4节各10分钟
4. 维度4：核心规则
- 足球：用脚控球，禁止手部触球（守门员除外），进球得1分
- 篮球：手脚并用控球，进球分2分球、3分球、罚球1分
<<|im_end|>
<<|im_start|>user
请写一篇文章，比较和对比足球和篮球两种运动，严格按照上述4个维度，用通顺的语言组织成段落，不得添加任何捏造的信息。
<<|im_end|>
<<|im_start|>assistant"""
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=5000, temperature=0.1)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
