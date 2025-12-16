import json
import logging
import uuid

import asyncio
from collections import defaultdict
from datetime import datetime, timezone, timedelta

import aio_pika
import pika
from sqlalchemy import func

from sqlalchemy.future import select
from agent_work.database.database import get_db, User, Session, Message, MessageSummary
from typing import Dict, Optional, Set, List

from agent_work.datasummary.summary_service import SUMMARY_TRIGGER_THRESHOLD, update_or_create_summary, \
    get_session_conversation_stats
from agent_work.util.timer_wrapper_for_func import async_timer_with_mark

# 消息队列配置（与之前记忆管理智能体共用一个队列）
QUEUE_NAME = "conversation_history_queue"
RABBITMQ_URL = "amqp://guest:guest@localhost:5672/"
# 新增：对话级触发防抖缓存（key：session_id，value：最后一次触发校验的时间）
# 作用：同一会话的多条消息，2秒内仅触发一次摘要校验
trigger_debounce_cache = defaultdict(lambda: datetime.min)
DEBOUNCE_INTERVAL = timedelta(seconds=2)  # 防抖间隔（可根据实际调整）

# async_memory_writer.py 中维护事件队列和处理器
summary_event_queue = asyncio.Queue()
is_summary_processor_running = True
processing_sessions: Set[str] = set()  # 移到全局，方便关闭时访问
processed_sessions: Set[str] = set()  # 记录已处理的会话，避免重复处理
# 新增：记录队列中待处理的session_id（用于去重）
pending_summary_tasks: Set[str] = set()


async def process_summary_events():
    """
    持续消费摘要事件队列，批量处理摘要生成（聚合1秒内的重复会话，避免重复处理）
    """
    global is_summary_processor_running

    while is_summary_processor_running:
        try:
            # 1秒超时：避免队列空时无限阻塞
            session_id = await asyncio.wait_for(summary_event_queue.get(), timeout=1.0)
            # 步骤1：兜底去重（防抖+队列去重可能漏网的情况）
            if session_id in processing_sessions or session_id in processed_sessions:
                # 从待处理集合中移除（避免内存泄漏）
                pending_summary_tasks.discard(session_id)
                summary_event_queue.task_done()
                continue

            # 步骤2：标记为正在处理
            processing_sessions.add(session_id)
            print(f"【摘要处理器】开始处理会话 {session_id}")

            try:
                await asyncio.sleep(1.0)  # 聚合消息
                await update_or_create_summary(session_id)  # 执行摘要生成
                processed_sessions.add(session_id)  # 标记已处理
                print(f"【摘要处理器】会话 {session_id} 处理完成")
            except Exception as e:
                print(f"【摘要处理器】会话 {session_id} 处理失败：{str(e)}")
            finally:
                # 步骤3：清理标记
                processing_sessions.discard(session_id)  # 移除处理标记，标记任务完成
                pending_summary_tasks.discard(session_id)  # 关键：从待处理集合移除，避免后续无法重新触发
                summary_event_queue.task_done()

        except asyncio.TimeoutError:
            continue  # 队列空，继续循环
        except Exception as e:
            print(f"【摘要处理器】循环异常：{str(e)}")
            continue


# 应用关闭时的清理函数
async def shutdown_summary_processor():
    """
    优雅关闭摘要服务：
    1. 停止接收新任务；
    2. 等待队列中所有任务处理完毕；
    3. 等待正在处理的任务完成；
    4. 清理资源
    """
    global is_summary_processor_running
    print("【摘要服务】开始关闭，停止接收新任务...")

    # 1. 停止运行标志，不再接收新任务
    is_summary_processor_running = False
    # 清理缓存（避免内存泄漏）
    trigger_debounce_cache.clear()
    pending_summary_tasks.clear()
    # 2. 等待队列中所有任务处理完毕（最多等待30秒，避免无限阻塞）
    max_wait_time = 30.0
    start_time = asyncio.get_event_loop().time()
    while not summary_event_queue.empty():
        elapsed_time = asyncio.get_event_loop().time() - start_time
        if elapsed_time > max_wait_time:
            print(f"【摘要服务】警告：等待队列任务超时（{max_wait_time}秒），仍有 {summary_event_queue.qsize()} 个任务未处理")
            break
        await asyncio.sleep(0.5)  # 轮询检查队列状态

    # 3. 等待正在处理的任务完成（最多等待10秒）
    start_time = asyncio.get_event_loop().time()
    while processing_sessions:
        elapsed_time = asyncio.get_event_loop().time() - start_time
        if elapsed_time > 10.0:
            print(f"【摘要服务】警告：等待正在处理的任务超时（10秒），仍有 {len(processing_sessions)} 个任务在处理")
            break
        print(f"【摘要服务】等待 {len(processing_sessions)} 个正在处理的任务完成...")
        await asyncio.sleep(0.5)

    # 4. 清理资源
    processed_sessions.clear()
    print("【摘要服务】已完全关闭")


# -------------------------- 生产者：发送对话数据到队列 --------------------------
@async_timer_with_mark(mark_param_name="session_id")
async def send_message_to_queue_by_async(
        user_id: str,
        session_id: str,
        conversation_id: str,
        role: str,
        content: str,
        generate_time: str):
    connection = await aio_pika.connect_robust(RABBITMQ_URL)

    async with connection:
        channel = await connection.channel()
        queue = await channel.declare_queue(QUEUE_NAME, durable=True)

        # 为消息添加生产者生成的时间戳
        enriched_message = {
            "message_id": f"msg_{uuid.uuid4().hex[:8]}",  # 消息ID，也可由生产者生成
            "user_id": user_id,
            "session_id": session_id,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            # 核心：生产者生成的带时区的时间戳
            # "timestamp": datetime.now(timezone.utc).isoformat()
            "timestamp": generate_time
            # TODO 需要将该时间字段在真正的发送者侧设置【发送者可以先创建各种消息，待其业务逻辑完整结束后，发送消息即可。若发送者逻辑异常，可以丢弃因此产生的脏数据。如果业务上允许丢弃这种脏数据，或者将其存储到其他地方】，以使得消息的完整发送与存储
        }

        await channel.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps(enriched_message).encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT
            ),
            routing_key=queue.name
        )
        print(f"生产者发送消息: {enriched_message}")


# async_memory_writer.py 中调用，与事件驱动模型结合
async def trigger_summary_if_needed(session_id: str):
    """
    消息写入数据库后，检查是否需要触发摘要生成：
    未处理对话数 ≥5 时，触发摘要（放入事件队列异步处理）
    """
    # 对话级防抖：同一会话2秒内仅触发一次校验
    last_trigger_time = trigger_debounce_cache[session_id]
    if datetime.now() - last_trigger_time < DEBOUNCE_INTERVAL:
        # 2秒内已触发过，直接跳过
        print(f"【摘要触发】会话 {session_id} 触发防抖，2秒内已校验过，跳过")
        return
    # 更新最后触发时间
    trigger_debounce_cache[session_id] = datetime.now()
    # 获取会话的对话统计
    total_conv_count, _ = await get_session_conversation_stats(session_id)
    last_processed_count = 0
    async for db in get_db():
        summary_record = await db.execute(select(MessageSummary).filter_by(session_id=session_id))
        summary_record = summary_record.scalars().first()
        last_processed_count = summary_record.last_processed_conversation_count if summary_record else 0

    # 计算未处理对话数
    unprocessed_conv_count = total_conv_count - last_processed_count
    if unprocessed_conv_count >= SUMMARY_TRIGGER_THRESHOLD:
        # 队列任务去重：检查队列中是否已有该session_id的未处理任务
        # 注意：asyncio.Queue无公开的“查看队列内容”方法，通过临时集合记录待处理任务
        if session_id in pending_summary_tasks:
            print(f"【摘要触发】会话 {session_id} 已有任务在队列中，跳过重复入队")
            return
        print(f"【触发机制】会话 {session_id} 未处理对话数达到 {unprocessed_conv_count}，触发摘要生成...")
        # 放入事件队列异步处理（沿用之前的事件驱动模型）
        await summary_event_queue.put(session_id)
    else:
        print(
            f"【触发机制】会话 {session_id} 未处理对话数 {unprocessed_conv_count}，未达到触发阈值（{SUMMARY_TRIGGER_THRESHOLD}）")


def send_conversation_to_queue(
        user_id: str,
        session_id: str,
        conversation_id: str,
        role: str,
        content: str
):
    """
    发送对话数据到消息队列（在用户收到回复后调用）
    :param user_id: 用户ID（若暂无登录系统，可传默认值如"default_user"）
    :param session_id: 现有会话ID
    :param conversation_id: 现有对话ID
    :param role: 消息角色（user/retriever/expert）
    :param content: 消息内容
    """
    # 构造消息数据
    message_data = {
        "user_id": user_id,
        "session_id": session_id,
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
        "message_id": f"msg_{uuid.uuid4().hex[:8]}",  # 消息唯一ID
        # 核心：生产者生成的带时区的时间戳
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    # 连接消息队列并发送
    connection = pika.BlockingConnection(pika.URLParameters(url=RABBITMQ_URL))
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME, durable=True)  # 持久化队列
    channel.basic_publish(
        exchange='',
        routing_key=QUEUE_NAME,
        body=json.dumps(message_data),
        properties=pika.BasicProperties(delivery_mode=2)  # 持久化消息
    )
    connection.close()
    print(f"对话数据已发送到队列：{message_data}")


async def consume_conversation_queue():
    # 1. 异步连接 RabbitMQ
    connection = await aio_pika.connect_robust(RABBITMQ_URL)

    # 2. 创建通道
    channel = await connection.channel()

    # 3. 声明队列（与生产者一致）
    queue = await channel.declare_queue(QUEUE_NAME, durable=True)

    # 4. 定义异步回调函数（关键：回调是异步的，不阻塞事件循环）
    async def callback(message: aio_pika.IncomingMessage):
        async with message.process():  # 自动确认消息（处理完成后 ack）
            try:
                # 解析消息数据
                data = json.loads(message.body)
                print(f"收到消息：{data['message_id']}，开始写入数据库...")

                # 5. 异步调用数据库写入（事件循环可正常调度）
                await write_conversation_to_db(data)

                print(f"消息 {data['message_id']} 写入数据库成功！")

            except Exception as e:
                # 处理失败，消息会重新入队（需配置 RabbitMQ 重试策略）
                print(f"消息 {data.get('message_id')} 处理失败：{str(e)}")
                # 手动拒绝消息，让其重新入队（durable=True 时消息不会丢失）
                await message.reject(requeue=True)

    # 6. 开始异步消费（不阻塞事件循环，事件循环可同时处理多个消息）
    await queue.consume(callback)
    print("异步消费服务已启动，等待消息...")

    # 7. 保持连接（让事件循环持续运行）
    await asyncio.Future()  # 无限等待，直到程序被中断


async def write_conversation_to_db(data: Dict):
    # 初始化数据库连接
    async for db in get_db():
        try:
            """将对话数据写入数据库（核心逻辑）"""
            # 1. 确保用户存在（无用户系统时自动创建默认用户）
            result = await db.execute(select(User).filter_by(user_id=data["user_id"]))
            user = result.scalars().first()
            if not user:
                user = User(user_id=data["user_id"])
                db.add(user)
                await db.flush()  # 刷新获取用户ID

            # 2. 确保会话存在（关联用户）
            result = await db.execute(select(Session).filter_by(session_id=data["session_id"]))
            session = result.scalars().first()
            if not session:
                session = Session(session_id=data["session_id"], user_id=data["user_id"])
                db.add(session)
                await db.flush()

            # 3. 写入对话消息
            message = Message(
                id=data["message_id"],
                session_id=data["session_id"],
                conversation_id=data["conversation_id"],
                role=data["role"],
                content=data["content"],
                timestamp=datetime.fromisoformat(data["timestamp"])  # 将字符串日期转换为datetime实例
            )
            db.add(message)

            # !!! 修正后的逻辑：在提交前，获取当前消息总数 !!!
            # 注意：此时 message 还未被提交，所以 count 是当前总数（不包含新消息）
            result = await db.execute(
                select(func.count(Message.id)).filter_by(session_id=data["session_id"])
            )
            current_message_count = result.scalar_one()

            # 提交事务，将新消息写入数据库
            await db.commit()
            print(f"数据库写入成功：{data['message_id']}")
            # 4. 检查是否需要触发摘要
            await trigger_summary_if_needed(data["session_id"])
        except Exception as e:
            # 内部异常立即抛出，让外层 future.result() 捕获
            await db.rollback()  # 回滚事务
            print(f"数据库写入内部失败：{str(e)}")
            raise  # 重新抛出异常，触发回调的 basic_nack


async def main():
    """消费者服务启动入口：同时启动消息消费和摘要处理器"""
    # 启动摘要处理器协程
    summary_task = asyncio.create_task(process_summary_events())
    print("【摘要服务】已启动")

    # 启动消息消费协程
    consume_task = asyncio.create_task(consume_conversation_queue())
    print("【消费者服务】已启动，开始监听队列...")

    # 等待两个协程完成（或被中断）
    try:
        await asyncio.gather(summary_task, consume_task)
    except KeyboardInterrupt:
        print("\n【消费者服务】收到关闭信号...")
        await shutdown_summary_processor()
        await asyncio.gather(summary_task, consume_task, return_exceptions=True)


# 新增
# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AsyncPersistenceService:
    """异步持久化服务：消费Redis队列，落库+发MQ"""
    from agent_work.util.redis_util import AgentMessage

    def __init__(self, mq_client, msg_db, conv_db):
        self.mq_client = mq_client
        self.msg_db = msg_db
        self.conv_db = conv_db
        from agent_work.util.redis_util import redis_queue  # 全局redis消息队列实例
        self.redis_queue = redis_queue  # 注入Redis队列
        self.is_running = False
        self.MQ_MAX_RETRY = 3
        self.BATCH_SIZE = 10  # 批量处理大小

    async def start_persistence_task(self):
        """启动异步持久化任务（消费Redis队列）"""
        self.is_running = True
        logger.info("异步持久化任务已启动（消费Redis队列）")
        while self.is_running:
            try:
                # 批量消费Redis队列消息（阻塞式，避免空轮询）
                batch_messages = await self.redis_queue.batch_get_messages(
                    batch_size=self.BATCH_SIZE,
                    timeout=1
                )

                if not batch_messages:
                    await asyncio.sleep(0.1)
                    continue

                # 批量落库（复用原有逻辑）
                await self._batch_save_to_db(batch_messages)
                # 批量发MQ（复用原有逻辑）
                await self._batch_send_to_mq(batch_messages)

            except Exception as e:
                logger.error(f"异步持久化任务异常：{e}", exc_info=True)
                await asyncio.sleep(1)

    # 以下_batch_save_to_db/_batch_send_to_mq/stop_persistence_task逻辑完全复用，无需修改
    async def _batch_save_to_db(self, messages: List[AgentMessage]):
        # 复用原有批量落库逻辑...
        pass

    async def _batch_send_to_mq(self, messages: List[AgentMessage]):
        # 复用原有批量发MQ逻辑...
        pass

    async def stop_persistence_task(self):
        self.is_running = False
        logger.info("异步持久化任务已停止")


if __name__ == '__main__':
    # 单独运行，持续消费队列并写入数据库

    """
    asyncio.run() 打断点调试不生效，核心原因是 asyncio.run() 会启动独立事件循环并阻塞主线程，调试器无法 “穿透” 到异步任务内部—— 
    调试器默认跟踪主线程，但 consume_conversation_queue() 是在事件循环管理的异步任务中执行，而非主线程，导致断点 “看不到” 执行流程。
    """
    asyncio.run(main())
    # 调试策略：采用显式创建事件循环，构造具体方法入参，执行异步协程方法
    # 显式创建事件循环
    # loop = asyncio.get_event_loop()
    # try:
    #     # 运行主任务（调试器可穿透到 consume_conversation_queue 内部）
    #     loop.run_until_complete(consume_conversation_queue())
    #     # loop.run_until_complete(write_conversation_to_db({"user_id": "default_user", "session_id": "session_1dfc1ade", "conversation_id": "conversation_d51da412", "role": "user", "content": "润滑脂用什么型号？", "message_id": "msg_a2c977ab"}))
    # finally:
    #     # 关闭循环（可选，程序退出时自动关闭）
    #     loop.close()
