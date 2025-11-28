import json
import uuid

import aio_pika
import pika

from sqlalchemy.future import select
from agent_work.database.database import get_db, User, Session, Message
from typing import Dict, Optional

# 消息队列配置（与之前记忆管理智能体共用一个队列）
QUEUE_NAME = "conversation_history_queue"
RABBITMQ_URL = "amqp://guest:guest@localhost:5672/"


# -------------------------- 生产者：发送对话数据到队列 --------------------------
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
        "message_id": f"msg_{uuid.uuid4().hex[:8]}"  # 消息唯一ID
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


# -------------------------- 消费者：异步写入数据库 --------------------------
# async def consume_conversation_queue():
#     """消费队列中的对话数据，异步写入数据库"""
#     # 获取当前事件循环
#     loop = asyncio.get_running_loop()
#     # 连接消息队列
#     connection = pika.BlockingConnection(pika.URLParameters(url=RABBITMQ_URL))
#     channel = connection.channel()
#     channel.queue_declare(queue=QUEUE_NAME, durable=True)
#
#     def callback(ch, method, properties, body):
#         try:
#             # 解析消息数据
#             data = json.loads(body)
#             # 使用 asyncio.run_coroutine_threadsafe 将异步任务提交到事件循环
#             # 1. 提交异步写入任务，获取 Future 对象
#             future = asyncio.run_coroutine_threadsafe(write_conversation_to_db(data), loop)
#             # 2. 阻塞等待任务执行完成（设置超时，避免无限等待）
#             # 超时时间根据数据库写入耗时调整（如5秒）
#             future.result(timeout=5)
#             """
#             pika 库的回调函数是同步执行的，它运行在一个单独的线程中，而不是在 asyncio 的事件循环线程中。
#             当你在这个同步回调中调用 asyncio.run() 时，它会尝试创建一个新的事件循环，这与你已经启动的事件循环冲突。
#             """
#             # asyncio.run(write_conversation_to_db(db, data))
#             # 确认消息处理完成
#             ch.basic_ack(delivery_tag=method.delivery_tag)
#             print(f"对话数据写入成功：{data['message_id']}")
#         except asyncio.TimeoutError:
#             # 任务超时，拒绝消息并重新入队
#             print(f"写入超时，消息重新入队：{body}")
#             ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
#         except Exception as e:
#             # 处理失败，重新入队（最多重试3次）
#             ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
#             print(f"对话数据写入失败（重试）：{str(e)}")
#
#     # 开始消费队列（手动确认消息）
#     channel.basic_qos(prefetch_count=1)  # 公平分发，避免单消费者过载
#     channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback)
#     print("对话历史消费服务已启动，等待数据...")
#     channel.start_consuming()
# -------------------------- 消费者：异步消费（核心改造，用 aio_pika） --------------------------
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
                content=data["content"]
            )
            db.add(message)
            await db.commit()
        except Exception as e:
            # 内部异常立即抛出，让外层 future.result() 捕获
            await db.rollback()  # 回滚事务
            print(f"数据库写入内部失败：{str(e)}")
            raise  # 重新抛出异常，触发回调的 basic_nack


if __name__ == '__main__':
    # 单独运行，持续消费队列并写入数据库
    import asyncio
    """
    asyncio.run() 打断点调试不生效，核心原因是 asyncio.run() 会启动独立事件循环并阻塞主线程，调试器无法 “穿透” 到异步任务内部—— 
    调试器默认跟踪主线程，但 consume_conversation_queue() 是在事件循环管理的异步任务中执行，而非主线程，导致断点 “看不到” 执行流程。
    """
    asyncio.run(consume_conversation_queue())
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