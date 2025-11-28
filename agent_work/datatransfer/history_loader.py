from sqlalchemy.future import select
from agent_work.database.database import get_db, Session, Message
from typing import List, Dict


async def load_conversation_history(session_id: str, user_id: str) -> List[Dict]:
    """
    按会话ID加载历史对话（用户隔离）
    :return: 按时间戳排序的对话列表，格式：[{"role": "user", "content": "...", "timestamp": "..."}]
    """
    async for db in get_db():
        # 1. 验证会话归属（用户隔离，避免跨用户访问）
        result = await db.execute(
            select(Session).where(Session.session_id == session_id, Session.user_id == user_id)
        )
        session = result.scalars().first()
        if not session:
            return []  # 会话不存在或无权限

        # 2. 加载该会话的所有对话（按时间戳升序）
        result = await db.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.timestamp.asc())
        )
        messages = result.scalars().all()

        # 3. 格式化返回
        return [
            {
                "role": msg.role.value,
                "content": msg.content,
                "timestamp": msg.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            } for msg in messages
        ]


# 示例：在智能体处理对话前加载历史
async def inject_history_to_agent(agent, session_id: str, user_id: str):
    """将历史对话注入智能体内存"""
    history = await load_conversation_history(session_id, user_id)
    for msg in history:
        agent_msg = Msg(
            name=msg["role"],
            content=msg["content"],
            role=msg["role"],
            timestamp=msg["timestamp"]
        )
        await agent.memory.add(agent_msg)