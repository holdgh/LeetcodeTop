from openai import AsyncOpenAI
from typing import Optional

# LLM配置（可复用现有LLM客户端，或单独配置轻量模型）
LLM_CLIENT = AsyncOpenAI(api_key="your-api-key", base_url="your-base-url")
REWRITE_MODEL = "gpt-3.5-turbo"  # 轻量模型足够，成本更低
REWRITE_TEMPERATURE = 0.2  # 低温度，保证输出稳定


class QuestionRewriteAgent:
    """问题重写智能体：将当前问题结合历史上下文，重写为独立可解答的问题"""

    @staticmethod
    async def rewrite_question(session_id: str, current_question: str, history_context: str) -> str:
        """
        核心方法：重写问题
        参数：
            session_id: 会话ID（用于日志）
            current_question: 用户当前原始问题
            history_context: 历史上下文（摘要+最近对话）
        返回：重写后的问题
        """
        if not history_context:
            # 无历史上下文，直接返回原始问题
            print(f"【重写智能体】会话 {session_id} 无历史上下文，无需重写")
            return current_question

        # 构建Prompt（简洁明确，避免大模型发散）
        prompt = f"""
        你是问题重写助手，需要结合历史上下文，将用户当前问题重写为「独立可解答的完整问题」，要求：
        1. 包含历史上下文中的关键背景（如用户之前的需求、讨论的结论），但不冗余；
        2. 保留用户当前问题的核心诉求，不改变原意；
        3. 语言简洁、连贯，适合直接提交给问答智能体；
        4. 仅输出重写后的问题，不添加任何额外解释。

        历史上下文：
        {history_context}

        用户当前问题：
        {current_question}

        重写后的问题：
        """

        try:
            response = await LLM_CLIENT.chat.completions.create(
                model=REWRITE_MODEL,
                messages=[
                    {"role": "system", "content": "严格遵循上述规则，仅输出重写后的问题"},
                    {"role": "user", "content": prompt}
                ],
                temperature=REWRITE_TEMPERATURE,
                max_tokens=500,
                timeout=10  # 超时控制
            )
            rewritten_question = response.choices[0].message.content.strip()
            print(
                f"【重写智能体】会话 {session_id} 问题重写完成\n原始问题：{current_question}\n重写后：{rewritten_question}")
            return rewritten_question
        except Exception as e:
            print(f"【重写智能体】会话 {session_id} 问题重写失败：{str(e)}")
            # 降级策略：重写失败时返回原始问题，不影响主流程
            return current_question


# 单例实例（避免重复初始化）
rewrite_agent = QuestionRewriteAgent()
