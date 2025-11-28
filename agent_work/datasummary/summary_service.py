import asyncio
import datetime
import pytz

from agentscope.model import DashScopeChatModel
from sqlalchemy import func
from sqlalchemy.future import select
from agent_work.database.database import get_db, Message, MessageSummary
from agent_work.util.dashscope_response_parser import parse_model_response

# 大模型实例
summary_model = DashScopeChatModel(
    model_name="deepseek-v3",
    api_key="sk-f61034a0afd64ffdab4be83a063b20e3",
    # api_key=os.getenv("DASHSCOPE_API_KEY"),
    # temperature=0.1  # 降低随机性，保证运维回答准确性
    generate_kwargs={
        "temperature": 0.1,
        "top_p": 0.8,
        "max_tokens": 300,
        "repetition_penalty": 1.1
    }
)

MAX_TOKEN_LIMIT = 12000  # 预留 4k tokens 给模型输出（16k 模型总上限）
SUMMARY_TRIGGER_THRESHOLD = 10  # 每 10 条消息增量更新一次


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


async def generate_incremental_summary(session_id: str) -> tuple:
    """增量生成摘要：上一次摘要 + 新增消息"""
    async for db in get_db():
        # 1. 查询会话的摘要记录（获取上一次摘要和截止消息ID）
        summary_result = await db.execute(select(MessageSummary).filter_by(session_id=session_id))
        summary_record = summary_result.scalars().first()

        # 2. 查询新增的消息（上一次摘要之后的消息）
        if summary_record and summary_record.summary_time:
            # 增量更新：只查 last_summarized_msg_id 之后的消息
            msg_result = await db.execute(
                select(Message)
                .filter_by(session_id=session_id)
                .where(Message.timestamp > summary_record.summary_time)  # 仅获取上次生成摘要时间之后消息
                .order_by(Message.timestamp)
            )
            new_messages = msg_result.scalars().all()
            previous_summary = summary_record.latest_summary or "该会话之前无重要内容。"
        else:
            # 首次摘要：查询前 SUMMARY_TRIGGER_THRESHOLD 条消息
            msg_result = await db.execute(
                select(Message)
                .filter_by(session_id=session_id)
                .order_by(Message.timestamp)
                .limit(SUMMARY_TRIGGER_THRESHOLD)
            )
            new_messages = msg_result.scalars().all()
            previous_summary = "该会话暂无历史摘要。"

        if not new_messages:
            return previous_summary  # 无新增消息，返回上次摘要

        # 3. 构建 Prompt（历史摘要 + 新增消息）
        prompt_messages = [
            {
                "role": "system",
                "content": """你是对话摘要助手，需要生成简洁、连贯的增量摘要。
规则：
1. 基于「历史摘要」和「新增消息」，合并生成新摘要（不是单独总结新增消息）；
2. 保留用户核心需求、关键问题、智能体的核心回复；
3. 忽略重复、无关的细节，控制摘要长度，避免冗余；
4. 用自然语言段落呈现，不要分点。"""
            },
            {
                "role": "user",
                "content": f"历史摘要：{previous_summary}"
            },
            {
                "role": "user",
                "content": "新增对话内容："
            }
        ]

        # 4. 拼接新增消息（并截断超长消息）
        total_tokens = estimate_tokens(previous_summary) + estimate_tokens("历史摘要和新增对话内容的提示文本")

        for msg in new_messages:
            truncated_content = truncate_long_text(msg.content)
            msg_tokens = estimate_tokens(truncated_content)

            # 检查总 tokens 是否超上限，超了则停止添加（优先保留早的消息）
            if total_tokens + msg_tokens > MAX_TOKEN_LIMIT:
                prompt_messages.append({
                    "role": "system",
                    "content": "（注：部分新增消息因长度限制未纳入摘要，核心内容已保留）"
                })
                break

            # 转换角色并添加到 Prompt
            llm_role = "assistant" if msg.role in ["expert", "retriever", "system"] else msg.role
            prompt_messages.append({
                "role": llm_role,
                "content": truncated_content
            })

            total_tokens += msg_tokens

        # 5. 调用 LLM 生成摘要
        try:
            response = await summary_model(
                messages=prompt_messages
            )
            new_summary = await parse_model_response(response)
            # 返回新摘要和本次处理的最后一条消息的时间戳
            if new_messages:
                latest_msg_timestamp = new_messages[-1].timestamp
            else:
                latest_msg_timestamp = None
            return new_summary, latest_msg_timestamp  # 返回新摘要和最后一条消息的发生时间
        except Exception as e:
            print(f"生成增量摘要失败：{str(e)}")
            raise Exception(f"Summary generation failed: {str(e)}")


async def update_or_create_summary(session_id: str):
    """更新或创建会话摘要（含增量逻辑）"""
    try:
        new_summary, latest_msg_timestamp = await generate_incremental_summary(session_id)
    except Exception as e:
        print(f"更新摘要失败：{str(e)}")
        return

    async for db in get_db():
        # 1. 查询当前消息总数
        msg_count_result = await db.execute(
            select(func.count(Message.id)).filter_by(session_id=session_id)
        )
        total_messages = msg_count_result.scalar_one()

        # 2. 更新或创建摘要记录
        summary_result = await db.execute(select(MessageSummary).filter_by(session_id=session_id))
        summary_record = summary_result.scalars().first()

        if summary_record:
            summary_record.latest_summary = new_summary
            summary_record.summary_time = latest_msg_timestamp
            summary_record.total_messages = total_messages
        else:
            summary_record = MessageSummary(
                session_id=session_id,
                latest_summary=new_summary,
                total_messages=total_messages,
                # 如果是首次摘要，summary_time 设为最新消息的时间戳
                summary_time=latest_msg_timestamp or datetime.datetime.now(pytz.UTC)
            )
            db.add(summary_record)

        await db.commit()
        print(f"会话 {session_id} 摘要更新成功（当前消息数：{total_messages}）")


async def _get_message_count(db, session_id: str) -> int:
    """
    辅助函数：获取指定会话的消息总数。
    """
    result = await db.execute(
        select(func.count(Message.id)).filter_by(session_id=session_id)
    )
    return result.scalar_one()


if __name__ == '__main__':
    # 调试策略：采用显式创建事件循环，构造具体方法入参，执行异步协程方法
    # 显式创建事件循环
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(update_or_create_summary('session_5dfc1ade'))
    finally:
        # 关闭循环（可选，程序退出时自动关闭）
        loop.close()
