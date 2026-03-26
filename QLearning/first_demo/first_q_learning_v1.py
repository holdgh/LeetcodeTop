#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2026/1/23 14:55
# @Author  : gaohuan
# @Email   : 
# @FileName: first_q_learning_v1.py
# @Desc    : 核心目标是训练一个基于多层感知机（MLP）的 “策略网络”，让智能体（Agent）学会控制小车平衡杆子，最终实现 “每轮累积奖励最大化”。

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import matplotlib.pyplot as plt


# ====================== 1. 策略网络定义（极简 MLP） ======================
class PolicyNetwork(nn.Module):  # 定义并初始化 “策略网络”（替代微调的 “加载 BERT/ERNIE 预训练模型”），策略网络是智能体的 “决策大脑”。
    """
    模型结构（2 层 MLP）：
        输入层：nn.Linear(state_dim, hidden_dim)（4 维状态→64 维隐藏层）；
        激活层：nn.ReLU()（引入非线性，拟合复杂决策规则）；
        输出层：nn.Linear(hidden_dim, action_dim)（64 维隐藏层→2 维动作概率）；
        输出处理：torch.softmax(logits, dim=-1)（将输出转为概率分布，满足 “动作概率和为 1”）。
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        logits = self.fc2(x)
        # 返回动作的概率分布（softmax）
        return torch.softmax(logits, dim=-1)


# ====================== 2. 强化学习核心训练类（GRPO/GSPO） ======================
class GRPO_GSPO_Trainer:
    def __init__(self, env_name="CartPole-v1", algorithm="grpo", lr=1e-3, lambda_grpo=0.01):
        # 初始化环境（关闭环境检查，避免bool8报错）
        # 环境与数据准备（对应微调的 “数据集准备”）。OpenAI Gym 提供的经典控制环境，目标是让小车保持杆子平衡
        self.env = gym.make(env_name, disable_env_checker=True)
        """
        记录环境的核心属性：
            state_dim=4：环境状态维度（小车位置、小车速度、杆子角度、杆子角速度）；
            action_dim=2：离散动作空间（向左推小车 / 向右推小车）。
        """
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.algorithm = algorithm  # 选择算法：grpo/gspo
        self.lambda_grpo = lambda_grpo  # GRPO 正则化系数

        # 初始化策略网络和优化器
        self.policy = PolicyNetwork(self.state_dim, self.action_dim)  # 模型实例化
        """
        优化器初始化：
            选择 Adam 优化器（强化学习主流选择，自适应学习率）；
            学习率lr=1e-3（CartPole 环境的适配值，复杂环境可调小至 1e-4）；
            优化目标：调整策略网络的权重（参数θ），让累积奖励最大化。
        """
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)  # 优化器初始化

        # 记录训练奖励（用于绘图）
        self.reward_history = []

    # 采样一条轨迹（状态、动作、奖励）- 核心修复：适配env.step返回值
    def sample_trajectory(self, max_steps=200):  # 轨迹采集，相当于数据准备和数据预处理
        states, actions, rewards = [], [], []
        # 适配不同gym版本的reset返回值
        reset_result = self.env.reset()  # 调用 env.reset() 重置环境，获取初始状态
        if isinstance(reset_result, tuple):
            state = reset_result[0]  # gym 0.26+ 返回 (state, info)
        else:
            state = reset_result  # 旧版gym仅返回state
        done = False
        step = 0
        # 循环执行 “预测动作→执行动作→保存数据”
        while not done and step < max_steps:  # 杆子倒下（任务失败）或步数超过 200（任务成功）
            # 转换状态为张量
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            # 预测动作概率
            action_probs = self.policy(state_tensor)  # 前向传播
            # 随机采样动作（策略梯度的核心：随机策略）
            action = np.random.choice(self.action_dim, p=action_probs.detach().numpy()[0])

            # 执行动作 - 核心修复：适配不同gym版本的step返回值
            step_result = self.env.step(action)  # 调用 env.step(action) 执行动作，获取新状态、奖励、终止标志
            if len(step_result) == 5:
                # gym 0.26+ 返回 (next_state, reward, terminated, truncated, info)
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated  # 合并终止条件
            else:
                # 旧版gym返回 (next_state, reward, done, info)
                next_state, reward, done, _ = step_result

            # 保存数据 保存每一步的state（状态）、action（动作）、reward（奖励）
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            step += 1

        # 计算累积奖励（折扣因子 γ=1，简单环境无需折扣）。作为策略优化的 “目标信号”（奖励越高，说明当前策略越好）
        total_reward = sum(rewards)  # 直接求和得到total_reward（CartPole 简单环境，无需折扣 / 标准化；复杂环境需加折扣因子γ和奖励标准化）
        self.reward_history.append(total_reward)

        # 转换为张量 将采集到的原始轨迹数据（list 格式）转换为模型可计算的张量格式，为损失计算做准备。
        states_tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.int64)
        # 一条完整的 “轨迹数据”（states, actions, rewards）+ 该轨迹的总奖励（累积奖励）。
        return states_tensor, actions_tensor, rewards, total_reward

    # 计算策略梯度损失（基础）
    def compute_policy_loss(self, states, actions, rewards):  # 基础策略梯度损失
        # 预测动作概率
        action_probs = self.policy(states)  # 前向传播
        # 取出选中动作的概率
        selected_probs = action_probs[range(len(actions)), actions]
        # 策略梯度损失：-E[log(p(a|s)) * R]（最大化奖励，所以加负号）
        policy_loss = -torch.mean(torch.log(selected_probs) * sum(rewards))
        return policy_loss

    # GRPO 损失（基础损失 + 梯度正则化）
    def compute_grpo_loss(self, states, actions, rewards):  # GRPO 损失（基础损失 + 梯度正则化）
        # 基础策略梯度损失
        policy_loss = self.compute_policy_loss(states, actions, rewards)

        # 计算策略梯度的 L2 范数（正则化项）
        grads = torch.autograd.grad(policy_loss, self.policy.parameters(), create_graph=True)
        grad_norm = sum(torch.sum(g ** 2) for g in grads)

        # GRPO 总损失
        grpo_loss = policy_loss + self.lambda_grpo * grad_norm
        return grpo_loss

    # GSPO 梯度更新（用梯度符号替代原始梯度）
    def update_with_gspo(self, states, actions, rewards):
        # 计算基础损失
        policy_loss = self.compute_policy_loss(states, actions, rewards)
        # 计算原始梯度
        self.optimizer.zero_grad()
        policy_loss.backward()

        # 替换梯度为符号（仅保留方向）
        for param in self.policy.parameters():
            if param.grad is not None:
                param.grad = torch.sign(param.grad)  # 采用模型参数的梯度符号作为其参数优化时的梯度值，例如若某模型参数的梯度为8.3797e-03，则此步后，其梯度为1.【torch.sign函数功能：正数值为1，负数值为-1，0值为0】【相当于舍弃了梯度的幅度】

        # 执行梯度下降
        self.optimizer.step()

    # 单轮训练
    def train_step(self):
        # 采样轨迹
        states, actions, rewards, total_reward = self.sample_trajectory()

        # 根据算法选择更新方式
        if self.algorithm == "grpo":
            self.optimizer.zero_grad()
            loss = self.compute_grpo_loss(states, actions, rewards)
            loss.backward()
            self.optimizer.step()
        elif self.algorithm == "gspo":
            self.update_with_gspo(states, actions, rewards)

        return total_reward

    # 完整训练流程
    def train(self, epochs=200):
        print(f"开始训练 {self.algorithm.upper()} 算法，共 {epochs} 轮...")
        for epoch in range(epochs):
            total_reward = self.train_step()
            # 每 10 轮打印一次进度
            if (epoch + 1) % 10 == 0:
                avg_reward = np.mean(self.reward_history[-10:])
                print(f"Epoch {epoch + 1:3d} | 单轮奖励：{total_reward:3.0f} | 近10轮平均：{avg_reward:3.0f}")
            # CartPole-v1 收敛条件：平均奖励 ≥ 195（连续10轮）
            if len(self.reward_history) >= 10 and np.mean(self.reward_history[-10:]) >= 195:
                print(f"\n{self.algorithm.upper()} 训练收敛！Epoch: {epoch + 1}")
                break

    # 绘制奖励曲线
    def plot_rewards(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.reward_history, label=f"{self.algorithm.upper()} 奖励曲线")
        plt.axhline(y=195, color='r', linestyle='--', label="收敛阈值（195）")
        plt.xlabel("训练轮数")
        plt.ylabel("每轮累积奖励")
        plt.title(f"{self.algorithm.upper()} 在 CartPole-v1 上的训练曲线")
        plt.legend()
        plt.grid(True)
        plt.show()

    # 测试训练好的策略
    def test(self, episodes=5):
        print(f"\n测试 {self.algorithm.upper()} 策略（{episodes} 轮）...")
        total_rewards = []
        for ep in range(episodes):
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple):
                state = reset_result[0]
            else:
                state = reset_result
            done = False
            reward_ep = 0
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action_probs = self.policy(state_tensor)
                # 测试时选概率最大的动作（确定性策略）
                action = torch.argmax(action_probs).item()
                step_result = self.env.step(action)
                if len(step_result) == 5:
                    next_state, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    next_state, reward, done, _ = step_result
                state = next_state
                reward_ep += reward
            total_rewards.append(reward_ep)
            print(f"测试轮 {ep + 1}：奖励 = {reward_ep}")
        print(f"测试平均奖励：{np.mean(total_rewards):.1f}")


# ====================== 3. 主函数（跑通 GRPO/GSPO） ======================
if __name__ == "__main__":
    # ========== 可选1：训练 GRPO ==========
    # trainer_grpo = GRPO_GSPO_Trainer(algorithm="grpo", lr=1e-3, lambda_grpo=0.01)
    # trainer_grpo.train(epochs=200)
    # trainer_grpo.plot_rewards()
    # trainer_grpo.test(episodes=5)

    # ========== 可选2：训练 GSPO（取消注释即可运行） ==========
    trainer_gspo = GRPO_GSPO_Trainer(algorithm="gspo", lr=1e-3)
    trainer_gspo.train(epochs=200)
    trainer_gspo.plot_rewards()
    trainer_gspo.test(episodes=5)