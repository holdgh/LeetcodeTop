import asyncio
from typing import List, Tuple, Optional

from sqlalchemy import func, text
from sqlalchemy.future import select
from agent_work.database.database import get_db, Message, MessageSummary

# 配置参数（可根据大模型窗口调整）
RECENT_CONVERSATIONS_COUNT = 3  # 取最近3次对话的完整消息
CONTEXT_MAX_LENGTH = 2000  # 整合后的上下文最大长度（字符数）


async def get_session_history_context(session_id: str) -> str:
    """
    获取会话的历史上下文：全局摘要 + 最近3次对话消息
    返回：标准化的上下文文本（长度可控）
    """
    # 步骤1：并行查询摘要和最近对话（提升效率）
    summary_task = get_session_summary(session_id)
    recent_msgs_task = get_recent_conversation_messages(session_id, RECENT_CONVERSATIONS_COUNT)
    summary, recent_messages = await asyncio.gather(summary_task, recent_msgs_task)

    # 步骤2：构建标准化上下文
    context_parts = []

    # 2.1 全局摘要（优先添加，提供全局背景）
    if summary:
        context_parts.append(f"【会话全局摘要】\n{summary}\n")

    # 2.2 最近N次对话消息（补充细节，按时间排序）
    if recent_messages:
        context_parts.append(f"【最近{len(recent_messages)}次对话详情】\n")
        for msg in recent_messages:
            msg_time = msg.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            context_parts.append(f"[{msg_time}] {msg.role}：{msg.content}\n")

    # 步骤3：拼接并截断（避免超出大模型窗口）
    full_context = "".join(context_parts).strip()
    if len(full_context) > CONTEXT_MAX_LENGTH:
        # 截断时优先保留“最近对话”（细节更重要），再保留“摘要核心”
        recent_part = "".join(context_parts[1:])  # 最近对话部分
        summary_part = context_parts[0] if context_parts else ""  # 摘要部分
        # 先保留最近对话，剩余长度分配给摘要
        remaining_length = CONTEXT_MAX_LENGTH - len(recent_part)
        if remaining_length > 0:
            full_context = recent_part + "\n【会话全局摘要（部分）】\n" + summary_part[:remaining_length - 20] + "..."
        else:
            full_context = recent_part[:CONTEXT_MAX_LENGTH - 20] + "..."

    return full_context


async def get_session_summary(session_id: str) -> Optional[str]:
    """获取会话的最新全局摘要"""
    async for db in get_db():
        summary_record = await db.execute(
            select(MessageSummary.latest_summary)
            .filter_by(session_id=session_id)
            .order_by(MessageSummary.summary_time.desc())  # 取最新的摘要
        )
        summary = summary_record.scalars().first()
        return summary.strip() if summary else None


async def get_recent_conversation_messages(session_id: str, limit: int) -> List[Message]:
    """
    获取会话最近N次对话的所有消息（按时间戳排序）
    逻辑：先按对话起始时间取最近N个对话，再获取这些对话的所有消息并全局排序
    """
    async for db in get_db():
        # 步骤1：按对话起始时间取最近N个对话ID
        conversation_first_msg = (
            select(
                Message.conversation_id,
                func.min(Message.timestamp).label("first_msg_time")
            )
            .filter_by(session_id=session_id)
            .group_by(Message.conversation_id)
            .order_by(text("first_msg_time DESC"))  # 最近的对话在前
            .limit(limit)
            .subquery()
        )

        recent_conv_ids = await db.execute(select(conversation_first_msg.c.conversation_id))
        recent_conv_ids = [row[0] for row in recent_conv_ids.all()]
        if not recent_conv_ids:
            return []

        # 步骤2：获取这些对话的所有消息，按消息时间戳全局排序
        recent_messages = await db.execute(
            select(Message)
            .filter_by(session_id=session_id)
            .where(Message.conversation_id.in_(recent_conv_ids))
            .order_by(Message.timestamp.asc())
        )
        return recent_messages.scalars().all()


if __name__ == '__main__':
    result = asyncio.run(get_session_history_context('session_1dfc1ade'))
    print(result)
