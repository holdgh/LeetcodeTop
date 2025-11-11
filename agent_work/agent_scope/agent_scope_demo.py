# 基于agentscope实现双智能体协作

import asyncio
import os
from agentscope.agent import ReActAgent
from agentscope.message import Msg
from agentscope.pipeline import MsgHub
from agentscope.model import DashScopeChatModel
from agentscope.formatter import DashScopeMultiAgentFormatter


# 第一步：配置模型并创建智能体
def create_agent(name: str, background: str) -> ReActAgent:
    """创建并返回一个配置好的智能体。"""
    return ReActAgent(
        name=name,
        # system prompt 定义了智能体的角色和背景
        sys_prompt=f"你叫{name}。{background}在对话中请基于你的背景进行回复，每次发言尽量简洁。",
        model=DashScopeChatModel(
            model_name="qwen-max",  # 以通义千问模型为例，你可替换为其他支持的模型
            api_key="sk-f61034a0afd64ffdab4be83a063b20e3"
            # api_key=os.getenv("DASHSCOPE_API_KEY"),  # 请从环境变量读取你的API Key
        ),
        # 多智能体格式化器，用于处理多智能体对话的提示词构建
        formatter=DashScopeMultiAgentFormatter(),
    )


# 创建两个具备不同背景的智能体
alice = create_agent("Alice", "你是一位乐观开朗的市场专员，喜欢与人交流，对新兴科技充满兴趣。")
bob = create_agent("Bob", "你是一位逻辑严谨的软件工程师，做事喜欢追根究底，注重细节。")


# 第二步：使用 MsgHub 组织智能体协作
async def main():
    """运行智能体协作对话。"""

    # 使用 MsgHub 创建一个讨论组，包含 Alice 和 Bob
    async with MsgHub(
            participants=[alice, bob],  # 参与者列表
            # 讨论开始的公告消息，会广播给所有参与者
            announcement=Msg(
                "user",
                "大家好！我们现在开始讨论。今天的话题是：'AI 智能体在未来日常生活中可能扮演哪些角色？' 请分享一下你的看法。",
                "user",
            ),
    ):
        # 让每个智能体依次发言，进行3轮对话
        for _ in range(3):
            await alice()  # Alice发言，消息会自动通过MsgHub广播给Bob
            await bob()  # Bob发言，消息会自动通过MsgHub广播给Alice


# 运行程序
if __name__ == "__main__":
    # os.environ.setdefault("DASHSCOPE_API_KEY", 'sk-f61034a0afd64ffdab4be83a063b20e3')
    asyncio.run(main())