#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2026/1/29 13:26
# @Author  : gaohuan
# @Email   : 
# @FileName: training_worker_cpu.py
# @Desc    :
# !/usr/bin/env python3
"""
训练进程脚本：负责接收采样数据并进行模型训练
通过ZeroMQ接收采样进程的数据，适配Windows CPU环境（移除DeepSpeed依赖）
"""

import time
import torch
import zmq
import yaml
import pickle
import threading
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from data_types import Gsm8kTasksDataset, Episode, Gsm8kZhTasksDataset
from utils import group_advantages, grpo_loss, gspo_loss, train_accuracy, get_batch_log_probs, sample_trajectory, \
    reward_function
import numpy as np
from peft import LoraConfig, get_peft_model
import swanlab


class TrainingWorker:
    def __init__(self, config: dict):
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.config = config  # 设置config属性，以便将配置信息同步至swanlab，便于记录实验配置
        # 移除GPU相关配置，固定为CPU
        self.pretrained_model_path = config["model"]["pretrained_model_path"]
        self.ref_model_path = config["model"]["ref_model_path"]
        # CPU环境建议使用float32，避免bfloat16/float16的精度问题
        self.dtype = dtype_map.get(config["model"]["dtype"], torch.float32)
        self.data_path = config["data"]["data_path"]
        self.max_gen_len = config["data"]["max_gen_len"]
        self.train_batch_size = config["data"]["train_batch_size"]
        self.sample_batch_size = config["data"]["sample_batch_size"]
        self.test_size = config["data"]["test_size"]
        self.test_batch_size = config["data"]["test_batch_size"]
        self.num_answers_per_question = config["data"]["num_answers_per_question"]
        self.num_questions_per_batch = self.train_batch_size // self.num_answers_per_question
        self.use_gspo = config["training"]["use_gspo"]
        self.eval_interval = config["training"]["eval_interval"]
        self.sync_interval = config["training"]["sync_interval"]
        self.zmq_data_port = config["communication"]["data_port"]
        self.ckpt_dir = Path(config["checkpoint"]["ckpt_dir"])
        self.ckpt_file = config["checkpoint"]["ckpt_file"]
        self.use_lora = config["lora"]["enabled"]
        self.lora_rank = config["lora"]["rank"]
        self.lora_alpha = config["lora"]["alpha"]
        self.lora_lr = float(config["lora"]["learning_rate"])
        self.lora_dropoutp = config["lora"]["dropout"]
        self.lora_adapter_dir = Path(config["lora"]["adapter_dir"])

        # 创建保存目录
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.lora_adapter_dir.mkdir(parents=True, exist_ok=True)

        # 全局设备固定为CPU
        self.device = torch.device("cpu")
        self.stop_event = threading.Event()

        # 初始化模型、优化器、ZMQ
        self.setup_model()
        self.setup_zmq()

    def setup_model(self):
        """初始化模型、Tokenizer、优化器和学习率调度器（移除DeepSpeed，原生PyTorch实现）"""
        print(f"开始初始化模型，设备：{self.device}，数据类型：{self.dtype}")
        # 初始化策略模型 - 移除sdpa（CPU不支持），指定device为cpu
        self.new_policy_model = AutoModelForCausalLM.from_pretrained(
            self.pretrained_model_path,
            dtype=self.dtype,
            device_map=self.device,  # 强制加载到CPU
            trust_remote_code=True  # 兼容部分自定义模型
        )
        # 初始化Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_path,
            padding_side='left',
            trust_remote_code=True
        )
        # 补充pad_token（部分模型默认无pad_token）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.new_policy_model.config.pad_token_id = self.tokenizer.eos_token_id

        # 启用LoRA配置（原逻辑保留）
        if self.use_lora:
            lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                target_modules="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj".split(","),
                lora_dropout=self.lora_dropoutp,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.new_policy_model = get_peft_model(self.new_policy_model, lora_config)
            # 打印LoRA可训练参数
            self.new_policy_model.print_trainable_parameters()

        # 训练模式 + 启用梯度检查点（减少CPU内存占用）
        self.new_policy_model.train()
        self.new_policy_model.requires_grad_(True)
        self.new_policy_model.gradient_checkpointing_enable()
        print("梯度检查点已启用（减少CPU内存占用）")

        # ==新增：SwanLab配置（仅主进程初始化，避免多进程重复）
        self.swanlab_exp = None
        # 从配置读取API Key
        swanlab_api_key = self.config["swanlab"]["api_key"]
        # 自动登录
        swanlab.login(api_key=swanlab_api_key)
        # 初始化实验，可自定义项目名、实验名
        self.swanlab_exp = swanlab.init(
            project="Q_learning_demo",  # 项目名称（可自定义）
            experiment_name=f"gspo_{time.strftime('%Y%m%d_%H%M%S')}",  # 实验名（带时间戳）
            config=self.config,
            logdir="./swanlab_logs"  # 日志保存路径
            # sync_interval=1  # 每秒同步一次日志，缓存不会累积
        )
        print("SwanLab实验初始化完成，开始记录训练数据...")

        # 原生PyTorch优化器 - 替换DeepSpeed优化器
        # 区分LoRA和全量训练的学习率
        lr = self.lora_lr if self.use_lora else 1e-5
        self.optimizer = AdamW(
            params=self.new_policy_model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )

        # 学习率调度器
        self.scheduler = StepLR(
            optimizer=self.optimizer,
            step_size=100,  # 每100步学习率衰减
            gamma=0.95  # 衰减系数
        )

        # 梯度裁剪参数（原逻辑保留）
        self.max_grad_norm = 1.0
        print("模型、Tokenizer、优化器初始化完成（原生PyTorch CPU版）")

    def setup_zmq(self):
        """初始化ZeroMQ通信（移除分布式逻辑，单进程直接连接）"""
        self.context = zmq.Context()
        # 数据接收socket（PULL模式）- 单进程直接连接，无主从区分
        self.data_receiver = self.context.socket(zmq.PULL)
        # 绑定本地端口，允许采样进程连接
        # self.data_receiver.bind(f"tcp://*:{self.zmq_data_port}")
        self.data_receiver.connect(f"tcp://127.0.0.1:{self.zmq_data_port}")
        # self.data_receiver.bind(f"tcp://127.0.0.1:{self.zmq_data_port}")
        # 设置接收超时，避免阻塞
        self.data_receiver.setsockopt(zmq.RCVTIMEO, 100)
        # print(f"ZeroMQ初始化完成，绑定数据接收端口：tcp://*:{self.zmq_data_port}")
        print(f"ZeroMQ初始化完成，绑定数据接收端口：tcp://127.0.0.1:{self.zmq_data_port}")

    def deserialize_episodes(self, serialized_data):
        """反序列化episodes数据（原逻辑完全保留）"""
        episodes = []
        for data in serialized_data:
            episode = Episode(
                prefix=data['prefix'],
                prefix_tokens=data['prefix_tokens'],
                prefix_token_ids=data['prefix_token_ids'],
                generated_token_ids=data['generated_token_ids'],
                whole_token_ids=data['whole_token_ids'],
                is_finished=data['is_finished'],
                text=data['text'],
                reward=data['reward'],
                reward_info=data['reward_info'],
                old_policy_log_probs=data['old_policy_log_probs'],
                ref_policy_log_probs=data['ref_policy_log_probs']
            )
            episodes.append(episode)
        return episodes

    def train_step(self, episodes):
        """执行一个训练步骤（替换DeepSpeed引擎为原生PyTorch逻辑，核心GRPO/GSPO损失不变）"""
        # 计算前缀长度（原逻辑保留）
        prefix_len = len(episodes[0].whole_token_ids) - len(episodes[0].generated_token_ids)

        # 构造批次张量 - 强制指定device为cpu
        batch_token_ids = torch.tensor(
            [episode.whole_token_ids for episode in episodes],
            dtype=torch.long,
            device=self.device
        )
        attention_mask = (batch_token_ids != self.tokenizer.pad_token_id).long().to(self.device)

        # 计算新策略的概率分布（原逻辑保留，模型直接调用）
        new_policy_log_probs = get_batch_log_probs(
            model=self.new_policy_model,  # 替换原model_engine为原生模型
            batch_token_ids=batch_token_ids,
            attention_mask=attention_mask,
            enable_grad=True
        )

        # 构造奖励和优势函数（原逻辑保留）
        rewards = torch.tensor([episode.reward for episode in episodes], dtype=self.dtype, device=self.device)
        advantages = group_advantages(
            rewards=rewards,
            num_answers_per_question=self.num_answers_per_question
        ).to(self.device)

        # 构造旧策略和参考策略的log_probs（原逻辑保留）
        ref_policy_log_probs = torch.tensor(
            np.array([episode.ref_policy_log_probs for episode in episodes]),
            dtype=self.dtype,
            device=self.device
        )
        old_policy_log_probs = torch.tensor(
            np.array([episode.old_policy_log_probs for episode in episodes]),
            dtype=self.dtype,
            device=self.device
        )

        # 计算GRPO/GSPO损失（原核心逻辑完全保留）
        if self.use_gspo:
            loss = gspo_loss(
                ref_policy_log_probs=ref_policy_log_probs,
                old_policy_log_probs=old_policy_log_probs,
                new_policy_log_probs=new_policy_log_probs,
                attention_mask=attention_mask,
                advantages=advantages,
                prefix_len=prefix_len
            )
        else:
            loss = grpo_loss(
                ref_policy_log_probs=ref_policy_log_probs,
                old_policy_log_probs=old_policy_log_probs,
                new_policy_log_probs=new_policy_log_probs,
                attention_mask=attention_mask,
                advantages=advantages,
                prefix_len=prefix_len
            )

        # 原生PyTorch反向传播和优化（替换DeepSpeed engine.step）
        self.optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播计算梯度
        # 梯度裁剪（防止梯度爆炸，原逻辑保留）
        torch.nn.utils.clip_grad_norm_(
            self.new_policy_model.parameters(),
            max_norm=self.max_grad_norm
        )
        self.optimizer.step()  # 优化器更新参数
        self.scheduler.step()  # 学习率调度器步进

        return loss.item()  # 返回标量损失值

    def evaluate(self):
        """模型评估（原逻辑保留，适配CPU）"""
        with torch.no_grad():
            self.new_policy_model.eval()  # 评估模式
            # 加载评估数据集
            test_dataset = Gsm8kZhTasksDataset(
                data_path=self.data_path,
                tokenizer=self.tokenizer,
                split="test",
                test_size=self.test_size
            )
            test_dataloader = DataLoader(
                test_dataset,
                shuffle=True,
                collate_fn=Gsm8kZhTasksDataset.collate_fn,
                batch_size=self.test_batch_size,
            )

            # 评估指标初始化
            success_num = 0
            format_success_num = 0
            answer_success_num = 0
            entropy_sum = 0.0

            # 遍历评估批次
            for batch in test_dataloader:
                episodes = sample_trajectory(
                    model=self.new_policy_model,
                    batch=batch,
                    tokenizer=self.tokenizer,
                    max_gen_len=self.max_gen_len,
                    num_answer_per_question=1,
                    reward_function=reward_function,
                    device=self.device,
                    dtype=self.dtype
                )
                # 统计奖励指标
                for episode in episodes:
                    if np.abs(episode.reward_info["format_reward"] - 1.25) < 1e-3:
                        format_success_num += 1
                    if np.abs(episode.reward_info["answer_reward"] - 1.0) < 1e-3:
                        answer_success_num += 1
                    if np.abs(episode.reward - 2.25) < 1e-3:
                        success_num += 1

                # 计算熵值（原逻辑保留）
                batch_token_ids = torch.tensor(
                    [episode.whole_token_ids for episode in episodes],
                    dtype=torch.long,
                    device=self.device
                )
                attention_mask = (batch_token_ids != self.tokenizer.pad_token_id).long()
                batch_logits = self.new_policy_model(
                    input_ids=batch_token_ids,
                    attention_mask=attention_mask
                ).logits
                batch_logits = batch_logits[:, :-1, :]
                batch_probs = torch.softmax(batch_logits, dim=-1)
                batch_log_probs = torch.log(batch_probs + 1e-12)
                batch_token_entropy = -torch.sum(batch_probs * batch_log_probs, dim=-1)
                batch_entropy = batch_token_entropy.mean(dim=-1)
                entropy_sum += batch_entropy.sum().item()

            # 计算评估指标
            success_rate = success_num / self.test_size
            format_success_rate = format_success_num / self.test_size
            answer_success_rate = answer_success_num / self.test_size
            entropy = entropy_sum / self.test_size
            self.new_policy_model.train()  # 切回训练模式
        return success_rate, format_success_rate, answer_success_rate, entropy

    def run(self):
        """主运行循环（移除分布式广播，单进程直接处理数据）"""
        print("=== GRPO训练进程启动（Windows CPU版）===")
        train_step = 0

        # 初始评估原始模型性能
        eval_start_time = time.time()
        accuracy, format_accuracy, answer_accuracy, entropy = self.evaluate()
        print(
            f"初始模型评估 - 格式准确率: {format_accuracy:.4f}, 回答准确率: {answer_accuracy:.4f}, 平均熵: {entropy:.4f}, 评估时间: {time.time() - eval_start_time:.2f}s")

        # ==新增：记录到SwanLab
        self.swanlab_exp.log({
            "eval/format_accuracy": format_accuracy,
            "eval/answer_accuracy": answer_accuracy,
            "eval/average_entropy": entropy,
            "eval/step": train_step
        })

        try:
            while not self.stop_event.is_set():
                try:
                    # 从ZMQ接收采样数据（单进程直接接收，无主从区分）
                    data = self.data_receiver.recv()
                    serialized_episodes = pickle.loads(data)
                    episodes = self.deserialize_episodes(serialized_episodes)
                    sample_batch_size = len(episodes)

                    # ==新增：记录当前批次采样数据的奖励分布
                    rewards = [episode.reward for episode in episodes]
                    self.swanlab_exp.log({
                        "sample/reward_mean": np.mean(rewards),
                        "sample/reward_max": np.max(rewards),
                        "sample/reward_min": np.min(rewards),
                        "sample/batch_size": len(episodes)  # 可选：记录批次大小
                    })

                    # 数据合法性检查（原逻辑保留）
                    assert sample_batch_size % self.num_answers_per_question == 0, \
                        f"批次大小{sample_batch_size}无法被{self.num_answers_per_question}整除"
                    sample_questions_per_batch = sample_batch_size // self.num_answers_per_question

                    # 打印训练信息
                    rewards = [round(episode.reward, 4) for episode in episodes[:5]]  # 只打印前5个奖励
                    print(
                        f"\n训练步骤{train_step} - 数据批次大小: {sample_batch_size}, 问题数: {sample_questions_per_batch}, 前5个奖励: {rewards}...")

                    # 执行训练步骤
                    loss = self.train_step(episodes)
                    train_step += 1
                    print(
                        f"训练步骤{train_step - 1}完成 - 损失值: {loss:.6f}, 当前学习率: {self.scheduler.get_last_lr()[0]:.8f}")

                    # ==新增：记录训练损失（仅主进程）
                    self.swanlab_exp.log({
                        "train/loss": loss.item(),  # 记录损失值
                        "train/step": train_step  # 记录当前训练步
                    })

                    # ==新增：记录学习率
                    lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.lora_lr
                    self.swanlab_exp.log({"train/learning_rate": lr})

                    # 定期保存模型（原逻辑保留，适配原生模型）
                    if train_step % self.sync_interval == 0:
                        save_start = time.time()
                        if self.use_lora:
                            self.new_policy_model.save_pretrained(self.lora_adapter_dir)
                            print(f"保存LoRA适配器至 {self.lora_adapter_dir}，耗时: {time.time() - save_start:.2f}s")
                        else:
                            output_file = self.ckpt_dir / self.ckpt_file
                            torch.save(self.new_policy_model.state_dict(), output_file)
                            print(f"保存全量模型至 {output_file}，耗时: {time.time() - save_start:.2f}s")

                    # 定期评估模型性能
                    if train_step % self.eval_interval == 0:
                        eval_start_time = time.time()
                        accuracy, format_accuracy, answer_accuracy, entropy = self.evaluate()
                        print(
                            f"第{train_step}步评估 - 格式准确率: {format_accuracy:.4f}, 回答准确率: {answer_accuracy:.4f}, 平均熵: {entropy:.4f}, 评估时间: {time.time() - eval_start_time:.2f}s")

                        # ==新增：记录到SwanLab
                        self.swanlab_exp.log({
                            "eval/format_accuracy": format_accuracy,
                            "eval/answer_accuracy": answer_accuracy,
                            "eval/average_entropy": entropy,
                            "eval/step": train_step
                        })

                except zmq.Again:
                    # 无数据时短暂休眠，降低CPU占用
                    time.sleep(0.01)
                    continue
                except AssertionError as e:
                    print(f"数据合法性检查失败: {e}，跳过当前批次")
                    continue
                except Exception as e:
                    print(f"训练步骤错误: {e}，跳过当前批次", exc_info=True)
                    continue

        except KeyboardInterrupt:
            print("\n训练进程收到中断信号（Ctrl+C），开始优雅退出...")
        except Exception as e:
            print(f"训练进程异常终止: {e}", exc_info=True)
        finally:
            self.cleanup()
            # 退出前保存最后一次模型
            print("退出前保存最新模型...")
            if self.use_lora:
                self.new_policy_model.save_pretrained(self.lora_adapter_dir)
            else:
                output_file = self.ckpt_dir / self.ckpt_file
                torch.save(self.new_policy_model.state_dict(), output_file)
            print("最新模型保存完成，训练进程已退出")

    def cleanup(self):
        """清理资源（原逻辑保留，适配单进程）"""
        print("开始清理训练进程资源...")
        self.stop_event.set()
        # 关闭ZMQ连接
        if hasattr(self, 'data_receiver') and self.data_receiver:
            self.data_receiver.close()
        if hasattr(self, 'context'):
            self.context.term()
        # 清空CUDA缓存（CPU环境无实际作用，保留兼容）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("资源清理完成")


def main():
    """主函数（移除DeepSpeed相关打印，适配CPU）"""
    # 加载配置文件
    config_path = "./config_cpu.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 创建训练进程实例
    worker = TrainingWorker(config=config)

    # 打印启动信息
    print(f"数据接收端口: {config["communication"]["data_port"]}")
    print(f"LoRA启用: {worker.use_lora}, 评估间隔: {worker.eval_interval}步, 模型保存间隔: {worker.sync_interval}步")
    print("训练进程初始化成功，等待采样进程数据...")

    # 启动训练
    worker.run()


if __name__ == "__main__":
    # 设置PyTorch CPU多线程优化（根据电脑核心数调整）
    # torch.set_num_threads(8)  # 建议设置为CPU物理核心数
    # torch.set_num_interop_threads(4)
    main()