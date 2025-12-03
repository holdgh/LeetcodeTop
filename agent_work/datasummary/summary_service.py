import asyncio
import datetime
from typing import Tuple, List

from agentscope.model import DashScopeChatModel
from sqlalchemy import func
from sqlalchemy.future import select
from agent_work.database.database import get_db, Message, MessageSummary
from agent_work.util.dashscope_response_parser import parse_model_response

# 大模型实例
summary_model = DashScopeChatModel(
    model_name="deepseek-v3",
    api_key="sk-f61034a0afd64ffdab4be83a063b20e3",
    generate_kwargs={
        "temperature": 0.1,
        "top_p": 0.8,
        "max_tokens": 300,
        "repetition_penalty": 1.1
    }
)

MAX_TOKEN_LIMIT = 12000  # 预留 4k tokens 给模型输出（16k 模型总上限）
# SUMMARY_TRIGGER_THRESHOLD = 10  # 每 10 条消息增量更新一次
SUMMARY_TRIGGER_THRESHOLD = 5  # 每5次新对话触发摘要


# 辅助函数：估算文本的 tokens 数量（粗略估算，1 token ≈ 0.75 个英文单词 / 2 个中文字符）
def estimate_tokens(text: str) -> int:
    # 中文占比高的场景，用这个公式更准确
    chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
    other_chars = len(text) - chinese_chars
    return int(chinese_chars * 0.5 + other_chars * 0.75)


# 辅助函数：截断超长文本（确保单条消息不超上限）
def truncate_long_text(text: str, max_tokens: int = 2000) -> str:
    estimated_tokens = estimate_tokens(text)
    if estimated_tokens <= max_tokens:
        return text
    # 按比例截断（预留 100 tokens 给结尾提示）
    truncate_ratio = (max_tokens - 100) / estimated_tokens
    truncated_text = text[:int(len(text) * truncate_ratio)]
    return truncated_text + "..." + "\n（注：消息内容过长，已截断核心部分）"


async def get_session_conversation_stats(session_id: str) -> Tuple[int, List[str]]:
    """
    获取会话内的对话统计：
    返回：(总对话数, 按时间排序的未处理对话ID列表)
    逻辑：
    1. 按对话分组，取每个对话的第一条消息时间戳（确定对话顺序）；
    2. 按对话时间戳升序排序（保证对话发生顺序）；
    3. 对比上次处理的对话数，筛选出未处理的对话（最新N个）。
    """
    async for db in get_db():
        # 子查询：按conversation_id分组，获取每个对话的第一条消息时间戳
        conversation_first_msg = (
            select(
                Message.conversation_id,
                func.min(Message.timestamp).label("first_msg_time")  # 对话的起始时间
            )
            .filter_by(session_id=session_id)
            .group_by(Message.conversation_id)
            .subquery()
        )

        # 主查询：按对话起始时间排序，获取所有对话ID（保证顺序正确）
        ordered_conversations = await db.execute(
            select(conversation_first_msg.c.conversation_id)
            .order_by(conversation_first_msg.c.first_msg_time.asc())  # 按对话发生时间升序
        )
        all_conv_ids = [row[0] for row in ordered_conversations.all()]
        total_conv_count = len(all_conv_ids)

        # 查询上次摘要已处理的对话数
        summary_record = await db.execute(
            select(MessageSummary).filter_by(session_id=session_id)
        )
        summary_record = summary_record.scalars().first()
        last_processed_count = summary_record.last_processed_conversation_count if summary_record else 0

        # 计算未处理的对话ID（最新的N个，N=触发阈值）
        unprocessed_conv_ids = all_conv_ids[last_processed_count: last_processed_count + SUMMARY_TRIGGER_THRESHOLD]

        return total_conv_count, unprocessed_conv_ids  # 总的会话数量，未处理的会话id


async def get_conversation_messages(session_id: str, conversation_ids: List[str]) -> List[Message]:
    """
    获取指定对话的所有消息，按消息时间戳全局排序（保证跨对话的消息顺序）
    """
    async for db in get_db():
        if not conversation_ids:
            return []

        # 查询指定对话的所有消息，按消息自带的timestamp升序排序
        messages = await db.execute(
            select(Message)
            .filter_by(session_id=session_id)
            .where(Message.conversation_id.in_(conversation_ids))
            .order_by(Message.timestamp.asc())  # 核心：按消息时间戳排序，保证顺序
        )
        return messages.scalars().all()


async def generate_incremental_summary(session_id: str) -> Tuple[str, int] or None:
    """
    增量生成摘要：历史摘要 + 最新5次对话的所有消息
    返回：(新摘要, 本次处理的对话数) 或 None（无新对话）
    """
    # 步骤1：获取未处理的对话统计
    total_conv_count, unprocessed_conv_ids = await get_session_conversation_stats(session_id)
    if not unprocessed_conv_ids:
        print(f"【摘要服务】会话 {session_id} 无未处理对话，跳过摘要生成")
        return None
    unprocessed_conv_count = len(unprocessed_conv_ids)

    # 步骤2：获取历史摘要
    previous_summary = "该会话暂无历史摘要。"
    async for db in get_db():
        summary_record = await db.execute(select(MessageSummary).filter_by(session_id=session_id))
        summary_record = summary_record.scalars().first()
        previous_summary = summary_record.latest_summary if (
                    summary_record and summary_record.latest_summary) else "该会话暂无历史摘要。"

    # 步骤3：获取未处理对话的所有消息（按时间戳排序）
    unprocessed_messages = await get_conversation_messages(session_id, unprocessed_conv_ids)
    if not unprocessed_messages:
        print(f"【摘要服务】会话 {session_id} 未处理对话无消息，跳过摘要生成")
        return None

    # 步骤4：构建LLM Prompt（按时间顺序拼接消息）
    prompt = f"""
    你是会话摘要助手，需要基于以下信息生成简洁、连贯的增量摘要：
    1. 历史摘要：{previous_summary}
    2. 最新{unprocessed_conv_count}次对话的消息（按时间顺序排列）：

    """
    for msg in unprocessed_messages:
        # 拼接消息：角色 + 时间 + 内容（时间戳保留到秒，提升可读性）
        msg_time = msg.timestamp.strftime("%Y-%m-%d %H:%M:%S")  # 格式化时间戳
        prompt += f"[{msg_time}] {msg.role}：{msg.content}\n"

    prompt += """
    摘要生成规则：
    - 保留核心信息：用户需求、智能体回复结论、工具调用结果；
    - 按时间顺序整合，避免重复，逻辑连贯；
    - 语言简洁，不冗余，长度控制在500字以内；
    - 忽略格式细节（如特殊符号、重复确认语），聚焦语义。
    """

    # 步骤5：调用LLM生成摘要（异步调用，不阻塞事件循环）
    try:
        response = await summary_model(
            messages=[
                {"role": "system", "content": "你是专业的会话摘要助手，严格遵循上述规则生成摘要"},
                {"role": "user", "content": prompt}
            ]
        )
        new_summary = await parse_model_response(response)
        print(f"【摘要服务】会话 {session_id} 摘要生成成功，处理了 {unprocessed_conv_count} 次对话")
        return new_summary, unprocessed_conv_count  # 新摘要，当前处理的会话数量
    except Exception as e:
        print(f"【摘要服务】生成摘要失败（会话 {session_id}）：{str(e)}")
        raise


async def update_or_create_summary(session_id: str):
    """
    存储或更新摘要到数据库：
    - 存在摘要记录：更新摘要、已处理对话数、更新时间；
    - 不存在摘要记录：创建新记录。
    """
    try:
        # 步骤1：生成增量摘要
        summary_result = await generate_incremental_summary(session_id)
        if not summary_result:
            return
        new_summary, processed_conv_count = summary_result

        async for db in get_db():
            # 步骤2：查询会话总消息数（用于统计，可选）
            total_msg_count = await db.execute(
                select(func.count(Message.id)).filter_by(session_id=session_id)
            )
            total_msg_count = total_msg_count.scalar_one()

            # 步骤3：查询或创建摘要记录
            summary_record = await db.execute(select(MessageSummary).filter_by(session_id=session_id))
            summary_record = summary_record.scalars().first()

            if summary_record:
                # 更新现有记录
                summary_record.latest_summary = new_summary
                summary_record.last_processed_conversation_count += processed_conv_count  # 累加处理的对话数
                summary_record.summary_time = datetime.datetime.now(datetime.timezone.utc)  # 摘要更新时间（当前UTC时间）
                summary_record.total_messages = total_msg_count
            else:
                # 创建新记录
                summary_record = MessageSummary(
                    session_id=session_id,
                    latest_summary=new_summary,
                    last_processed_conversation_count=processed_conv_count,
                    summary_time=datetime.datetime.now(datetime.timezone.utc),
                    total_messages=total_msg_count
                )
                db.add(summary_record)

            # 提交事务
            await db.commit()
            print(f"【摘要服务】会话 {session_id} 摘要已保存到数据库，当前已处理 {summary_record.last_processed_conversation_count} 次对话")

    except Exception as e:
        print(f"【摘要服务】更新摘要失败（会话 {session_id}）：{str(e)}")
        # 回滚事务
        async for db in get_db():
            await db.rollback()
        raise


async def _get_message_count(db, session_id: str) -> int:
    """
    辅助函数：获取指定会话的消息总数。
    """
    result = await db.execute(
        select(func.count(Message.id)).filter_by(session_id=session_id)
    )
    return result.scalar_one()


async def get_all_valid_conversations(session_id: str) -> List[str]:
    """
    获取会话内所有有效对话ID（已删除的对话已被过滤），按时间排序
    """
    async for db in get_db():
        # 子查询：获取每个有效对话的第一条消息时间戳
        valid_conversations = (
            select(
                Message.conversation_id,
                func.min(Message.timestamp).label("first_msg_time")
            )
            .filter_by(session_id=session_id)
            .group_by(Message.conversation_id)
            .subquery()
        )

        # 按对话时间排序，返回所有有效对话ID
        ordered_conv_ids = await db.execute(
            select(valid_conversations.c.conversation_id)
            .order_by(valid_conversations.c.first_msg_time.asc())
        )
        return [row[0] for row in ordered_conv_ids.all()]


async def rebuild_full_summary(session_id: str):
    """
    全量重建摘要：基于当前所有有效对话，重新生成完整摘要
    """
    print(f"【摘要重建】开始重建会话 {session_id} 的摘要（因对话删除触发）")
    try:
        # 步骤1：获取所有有效对话ID（按时间排序）
        all_valid_conv_ids = await get_all_valid_conversations(session_id)
        if not all_valid_conv_ids:
            # 无有效对话，清空摘要
            async for db in get_db():
                summary_record = await db.execute(select(MessageSummary).filter_by(session_id=session_id))
                summary_record = summary_record.scalars().first()
                if summary_record:
                    await db.delete(summary_record)
                    await db.commit()
            print(f"【摘要重建】会话 {session_id} 无有效对话，已清空摘要")
            return

        # 步骤2：获取所有有效对话的消息（按时间戳排序）
        all_valid_messages = await get_conversation_messages(session_id, all_valid_conv_ids)
        if not all_valid_messages:
            async for db in get_db():
                summary_record = await db.execute(select(MessageSummary).filter_by(session_id=session_id))
                summary_record = summary_record.scalars().first()
                if summary_record:
                    await db.delete(summary_record)
                    await db.commit()
            print(f"【摘要重建】会话 {session_id} 无有效消息，已清空摘要")
            return

        # 步骤3：构建全量摘要Prompt
        prompt = f"""
        你是会话摘要助手，需要基于以下所有有效对话的消息（按时间顺序），生成完整的会话摘要：
        消息列表（按时间顺序排列）：

        """
        for msg in all_valid_messages:
            msg_time = msg.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            prompt += f"[{msg_time}] {msg.role}：{msg.content}\n"

        prompt += """
        摘要生成规则：
        - 保留所有核心信息：用户需求、智能体回复结论、工具调用结果；
        - 按时间顺序整合，逻辑连贯，避免重复；
        - 语言简洁，不冗余，长度控制在800字以内；
        - 忽略格式细节，聚焦语义完整性。
        """

        # 步骤4：调用LLM生成全量摘要
        response = await summary_model(
            messages=[
                {"role": "system", "content": "你是专业的会话摘要助手，严格遵循上述规则生成完整摘要"},
                {"role": "user", "content": prompt}
            ]
        )
        full_summary = parse_model_response(response)

        # 步骤5：更新数据库摘要（重置已处理对话数）
        async for db in get_db():
            total_msg_count = await db.execute(
                select(func.count(Message.id)).filter_by(session_id=session_id)
            )
            total_msg_count = total_msg_count.scalar_one()

            summary_record = await db.execute(select(MessageSummary).filter_by(session_id=session_id))
            summary_record = summary_record.scalars().first()

            if summary_record:
                # 更新现有记录
                summary_record.latest_summary = full_summary
                summary_record.last_processed_conversation_count = len(all_valid_conv_ids)  # 重置为当前有效对话总数
                summary_record.summary_time = datetime.datetime.now(datetime.timezone.utc)
                summary_record.total_messages = total_msg_count
            else:
                # 创建新记录
                summary_record = MessageSummary(
                    session_id=session_id,
                    latest_summary=full_summary,
                    last_processed_conversation_count=len(all_valid_conv_ids),
                    summary_time=datetime.datetime.now(datetime.timezone.utc),
                    total_messages=total_msg_count
                )
                db.add(summary_record)

            await db.commit()
        print(f"【摘要重建】会话 {session_id} 摘要重建成功，当前有效对话数：{len(all_valid_conv_ids)}")

    except Exception as e:
        print(f"【摘要重建】会话 {session_id} 摘要重建失败：{str(e)}")
        async for db in get_db():
            await db.rollback()
        raise


if __name__ == '__main__':
    # 调试策略：采用显式创建事件循环，构造具体方法入参，执行异步协程方法
    # 显式创建事件循环
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(update_or_create_summary('session_5dfc1ade'))
    finally:
        # 关闭循环（可选，程序退出时自动关闭）
        loop.close()
