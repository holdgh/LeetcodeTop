from sqlalchemy.future import select
from database import get_db, Message, MessageSummary
from agent_work.datasummary.summary_service import rebuild_full_summary  # 后续实现的全量重建函数


async def delete_conversation(session_id: str, conversation_id: str) -> bool:
    """
    删除指定会话下的某一次对话（含该对话的所有消息）
    返回：删除成功与否
    """
    async for db in get_db():
        try:
            # 步骤1：查询该对话的所有消息
            conversation_messages = await db.execute(
                select(Message)
                .filter_by(session_id=session_id, conversation_id=conversation_id)
            )
            conversation_messages = conversation_messages.scalars().all()
            if not conversation_messages:
                print(f"【对话删除】会话 {session_id} 下无对话 {conversation_id}，删除失败")
                return False

            # 步骤2：删除该对话的所有消息
            for msg in conversation_messages:
                await db.delete(msg)
            await db.commit()
            print(
                f"【对话删除】会话 {session_id} 下的对话 {conversation_id} 已删除（含 {len(conversation_messages)} 条消息）")

            # 步骤3：触发全量摘要重建（核心）
            await rebuild_full_summary(session_id)

            return True
        except Exception as e:
            await db.rollback()
            print(f"【对话删除】删除会话 {session_id} 的对话 {conversation_id} 失败：{str(e)}")
            return False
