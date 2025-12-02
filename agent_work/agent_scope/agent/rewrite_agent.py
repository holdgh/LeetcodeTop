from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeMultiAgentFormatter
from agentscope.model import ChatModelBase


def create_input_text_for_rewrite(current_question: str, history_context: str):
    return f"""
    历史上下文：
    {history_context}

    用户当前问题：
    {current_question}
"""


def create_rewrite_agent(chat_model: ChatModelBase) -> ReActAgent:
    rewrite = ReActAgent(
        name="问题重写助手",
        sys_prompt="""你是问题重写助手，需要结合历史上下文，将用户当前问题重写为「独立可解答的完整问题」，要求：
        1. 包含历史上下文中的关键背景（如用户之前的需求、讨论的结论），但不冗余；
        2. 保留用户当前问题的核心诉求，不改变原意；
        3. 语言简洁、连贯，适合直接提交给问答智能体；
        4. 仅输出重写后的问题，不添加任何额外解释。""",
        model=chat_model,
        formatter=DashScopeMultiAgentFormatter()
    )
    rewrite.set_console_output_enabled(False)  # 禁用控制台输出智能体内容，由业务代码控制输出内容
    return rewrite
