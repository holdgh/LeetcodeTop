import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import os
from typing import List, Dict, Optional
from agentscope.agent import ReActAgent
from agentscope.message import Msg, TextBlock, ToolUseBlock, ToolResultBlock
from agentscope.model import DashScopeChatModel, ChatModelBase
from agentscope.formatter import DashScopeMultiAgentFormatter


def create_expert_agent(chat_model: ChatModelBase) -> ReActAgent:
    expert = ReActAgent(
        name="运维专家",
        sys_prompt="""仅处理当前绑定会话的用户问题：
1. 基于专属内存中的用户问题和检索结果，按"原因→步骤→解决方案→注意事项"组织回复；
2. 仅关注当前会话信息，会话切换时内存会清空；
3. 多轮对话记住已提供内容，避免重复。""",
        model=chat_model,
        formatter=DashScopeMultiAgentFormatter()
    )
    expert.set_console_output_enabled(False)  # 禁用控制台输出智能体内容，由业务代码控制输出内容
    return expert
