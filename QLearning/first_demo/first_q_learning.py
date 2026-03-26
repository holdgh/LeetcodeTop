#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2026/1/23 14:27
# @Author  : gaohuan
# @Email   : 
# @FileName: first_q_learning.py
# @Desc    :
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import matplotlib.pyplot as plt


# ====================== 1. 策略网络定义（极简 MLP） ======================
class PolicyNetwork(nn.Module):
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
        # 初始化环境
        self.env = gym.make(env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.algorithm = algorithm  # 选择算法：grpo/gspo
        self.lambda_grpo = lambda_grpo  # GRPO 正则化系数

        # 初始化策略网络和优化器
        self.policy = PolicyNetwork(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # 记录训练奖励（用于绘图）
        self.reward_history = []

    # 采样一条轨迹（状态、动作、奖励）
    def sample_trajectory(self, max_steps=200):
        states, actions, rewards = [], [], []
        state = self.env.reset()[0]  # gym 0.26+ 返回 (state, info)
        done = False
        step = 0

        while not done and step < max_steps:
            # 转换状态为张量
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            # 预测动作概率
            action_probs = self.policy(state_tensor)
            # 随机采样动作（策略梯度的核心：随机策略）
            action = np.random.choice(self.action_dim, p=action_probs.detach().numpy()[0])

            # 执行动作
            next_state, reward, done, _, _ = self.env.step(action)

            # 保存数据
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            step += 1

        # 计算累积奖励（折扣因子 γ=1，简单环境无需折扣）
        total_reward = sum(rewards)
        self.reward_history.append(total_reward)

        # 转换为张量
        states_tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.int64)

        return states_tensor, actions_tensor, rewards, total_reward

    # 计算策略梯度损失（基础）
    def compute_policy_loss(self, states, actions, rewards):
        # 预测动作概率
        action_probs = self.policy(states)
        # 取出选中动作的概率
        selected_probs = action_probs[range(len(actions)), actions]
        # 策略梯度损失：-E[log(p(a|s)) * R]（最大化奖励，所以加负号）
        policy_loss = -torch.mean(torch.log(selected_probs) * sum(rewards))
        return policy_loss

    # GRPO 损失（基础损失 + 梯度正则化）
    def compute_grpo_loss(self, states, actions, rewards):
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
                param.grad = torch.sign(param.grad)

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
            state = self.env.reset()[0]
            done = False
            reward_ep = 0
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action_probs = self.policy(state_tensor)
                # 测试时选概率最大的动作（确定性策略）
                action = torch.argmax(action_probs).item()
                state, reward, done, _, _ = self.env.step(action)
                reward_ep += reward
            total_rewards.append(reward_ep)
            print(f"测试轮 {ep + 1}：奖励 = {reward_ep}")
        print(f"测试平均奖励：{np.mean(total_rewards):.1f}")


# ====================== 3. 主函数（跑通 GRPO/GSPO） ======================
if __name__ == "__main__":
    # ========== 可选1：训练 GRPO ==========
    trainer_grpo = GRPO_GSPO_Trainer(algorithm="grpo", lr=1e-3, lambda_grpo=0.01)
    trainer_grpo.train(epochs=200)
    trainer_grpo.plot_rewards()
    trainer_grpo.test(episodes=5)

    # ========== 可选2：训练 GSPO（取消注释即可运行） ==========
    # trainer_gspo = GRPO_GSPO_Trainer(algorithm="gspo", lr=1e-3)
    # trainer_gspo.train(epochs=200)
    # trainer_gspo.plot_rewards()
    # trainer_gspo.test(episodes=5)