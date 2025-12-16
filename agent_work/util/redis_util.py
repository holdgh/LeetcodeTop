import asyncio
import uuid
from datetime import datetime, timezone

import redis.asyncio as async_redis
import logging
import time
from dataclasses import dataclass, asdict
import json
from typing import Optional, List

from sqlalchemy import select

from agent_work.database.database import TempMessage, TempMessageStatus, get_db
from agent_work.datatransfer.async_memory_writer import process_summary_events, shutdown_summary_processor, \
    consume_conversation_queue

logger = logging.getLogger(__name__)


def create_msg_id() -> str:
    return f"msg_{uuid.uuid4().hex[:8]}"


# 消息数据结构（和之前一致，兼容序列化）
@dataclass
class AgentMessage:
    session_id: str
    user_id: str
    message_type: str  # user/rewrite/retrieve/expert
    content: str
    generate_time: str  # 消息生成时间戳
    message_id: str


async def fallback_save(msg: AgentMessage, status: str):
    """兜底落库：写入新增的temp_message表"""
    async for db in get_db():
        try:
            temp_msg = TempMessage(
                id=msg.message_id,
                session_id=msg.session_id,
                user_id=msg.user_id,
                message_type=msg.message_type,
                content=msg.content,
                generate_time=datetime.fromisoformat(msg.generate_time),
                backup_time=datetime.now(timezone.utc),  # 兜底写入时间
                status=status  # 临时消息数据状态 见枚举类TempMessageStatus定义
            )
            db.add(temp_msg)
            # 提交事务，将新消息写入数据库
            await db.commit()
            logger.warning(f"消息[{msg.session_id}]入队Redis失败，兜底写入temp_message表")
        except Exception as e:
            # 内部异常立即抛出，让外层 future.result() 捕获
            await db.rollback()  # 回滚事务
            print(f"数据库写入内部失败：{str(e)}")
            # 数据库也失败的极端场景：记录日志+告警，不中断接口
            logger.critical(f"兜底落库失败（数据库异常）：session_id={msg.session_id}, error={e}")
            # TODO 可选：触发告警（如钉钉/邮件）
            # await self.send_alert(f"兜底落库失败：{db_e}")
            raise  # 重新抛出异常，触发回调的 basic_nack


class RedisMessageQueue:
    """Redis分布式消息队列（替代内存队列，支持限流校验）"""

    def __init__(self, redis_host: str, redis_port: int, redis_db: int = 0, queue_key: str = "ops_qa_message_queue"):
        # 定义 IDE 提示缺失的配置属性
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        # 2. 用私有属性存储 Redis 客户端（避免递归调用）
        self._redis_client: Optional[async_redis.Redis] = None
        # 异步锁：防止并发初始化 Redis 客户端
        self._init_lock = asyncio.Lock()
        self.queue_key = queue_key
        # 队列阈值配置（可根据服务器性能调整）
        self.QUEUE_MAX_LENGTH = 10000  # 队列最大长度，超过则拒绝新请求
        self.QUEUE_WARN_LENGTH = 8000  # 队列告警阈值

    # @property
    # def redis_client(self):  # 注意方法名redis_client与私有属性_redis_client的区分。采用私有属性_redis_client是为了在访问属性redis_client时，避免递归调用
    #     """
    #     @property 递归调用问题
    #         @property
    #         def redis_client(self):
    #             if self.redis_client is None:  # 这里会再次调用 self.redis_client（@property 方法）
    #                 ...
    #             return self.redis_client
    #     上述代码中，self.redis_client is None 会触发对 @property 方法的递归调用，最终导致 RecursionError，这是比 IDE 提示更严重的运行时错误。
    #     """
    #     # 第一次访问时才加载文件
    #     if self._redis_client is None:
    #         try:
    #             self._redis_client = redis.Redis(
    #                 host=self.redis_host,
    #                 port=self.redis_port,
    #                 db=self.redis_db,
    #                 decode_responses=True  # 自动解码为字符串
    #             )
    #             logger.info("Redis客户端初始化成功")
    #         except Exception as e:
    #             logger.error(f"Redis客户端初始化失败：{e}")
    #             self._redis_client = None  # 标记为不可用
    #     return self._redis_client
    # 异步懒加载 Redis 客户端
    async def get_redis_client(self) -> Optional[async_redis.Redis]:
        """异步懒加载（替代 @property，避免 async property 兼容性问题）"""
        if self._redis_client is None:
            async with self._init_lock:  # 并发安全
                # 双重检查：防止并发初始化
                if self._redis_client is None:
                    try:
                        self._redis_client = async_redis.Redis(
                            host=self.redis_host,
                            port=self.redis_port,
                            db=self.redis_db,
                            decode_responses=True,
                            # 关键配置：解决Timeout reading问题
                            socket_timeout=15,  # 客户端等待响应超时（必须≥BRPOP的timeout）
                            socket_connect_timeout=5,  # 连接超时
                            socket_keepalive=True,  # 防止TCP连接断连
                            max_connections=50,  # 连接池大小，避免高并发耗尽
                        )
                        # 测试redis连接是否可用
                        await self._redis_client.ping()
                        logger.info(
                            f"异步Redis客户端初始化成功：{self.redis_host}:{self.redis_port}/{self.redis_db}")
                    except Exception as e:
                        logger.error(f"异步Redis客户端初始化失败：{e}", exc_info=True)
                        self._redis_client = None  # 将私有客户端属性置为空，后续再次获取时可再次触发实例化redis客户端逻辑，可实现重试功能
        return self._redis_client

    async def check_queue_threshold(self) -> bool:
        """
        校验队列长度：
        - 小于阈值：返回True（允许入队）
        - 大于等于阈值：返回False（拒绝入队）
        """
        client = await self.get_redis_client()
        if client is None:
            logger.error("异步Redis客户端未初始化，无法校验队列长度")
            return False
        try:
            queue_len = await client.llen(self.queue_key)
            if queue_len >= self.QUEUE_MAX_LENGTH:
                logger.warning(f"Redis队列已满（当前长度：{queue_len}，阈值：{self.QUEUE_MAX_LENGTH}），拒绝新请求")
                return False
            # 接近阈值时告警（可选）
            if queue_len >= self.QUEUE_WARN_LENGTH:
                logger.warning(f"Redis队列接近阈值（当前长度：{queue_len}，告警阈值：{self.QUEUE_WARN_LENGTH}）")
            return True
        except async_redis.TimeoutError as e:
            logger.warning(f"校验Redis消息队列超时：{e}")
            self._redis_client = None  # 将私有客户端属性置为空，后续再次获取时可再次触发实例化redis客户端逻辑，可实现重试功能
            return False

        except async_redis.ConnectionError as e:
            logger.error(f"Redis连接异常：{e}", exc_info=True)
            self._redis_client = None  # 将私有客户端属性置为空，后续再次获取时可再次触发实例化redis客户端逻辑，可实现重试功能
            return False

        # 其他异常，仅打印
        except Exception as e:
            logger.error(f"Redis校验队列长度失败：{e}", exc_info=True)
            return False

    async def put_message(self, msg: AgentMessage, retry_times: int = 2) -> bool:
        """
        消息入队（先校验阈值，再入队）
        :return: True=入队成功，False=队列满/入队失败
        """
        # 第一步：校验队列阈值
        if not await self.check_queue_threshold():
            logger.error("redis消息队列长度已达到最大值，无法向redis消息队列写入消息，启用补偿机制将数据存入临时表")
            # 入队失败兜底：直接落库（应急方案）
            await fallback_save(msg, TempMessageStatus.REDIS_FAILED)
            return False

        client = await self.get_redis_client()
        if client is None:
            logger.error("异步Redis客户端未初始化，无法向redis消息队列写入消息，启用补偿机制将数据存入临时表")
            # 入队失败兜底：直接落库（应急方案）
            await fallback_save(msg, TempMessageStatus.REDIS_FAILED)
            return False
        # 第二步：消息序列化（dataclass转JSON字符串）
        # 重试逻辑：仅针对客户端超时异常
        for retry in range(retry_times + 1):
            try:
                msg_json = json.dumps(asdict(msg))
                # LPUSH：从队列头部入队，BRPOP从尾部出队，保证FIFO
                await client.lpush(self.queue_key, msg_json)
                logger.debug(f"消息入队Redis成功：会话[{msg.session_id}]，类型[{msg.message_type}]")
                return True
            except async_redis.TimeoutError as e:
                if retry >= retry_times:
                    logger.error(f"Redis 写入数据超时（重试{retry_times}次后仍失败）：{e}", exc_info=True)
                    break
                logger.warning(f"Redis 写入数据超时，第{retry + 1}次重试...：{e}")
                # 重试前重置客户端（仅超时异常时重置）仅连接超时情况下重试
                self._redis_client = None
                client = await self.get_redis_client()
                if client is None:
                    logger.error("重试写入数据时Redis客户端初始化失败，终止重试")
                    break

            except async_redis.ConnectionError as e:
                logger.error(f"Redis连接异常：{e}", exc_info=True)
                self._redis_client = None
                break
            except Exception as e:
                logger.error(f"消息入队Redis失败：{e}", exc_info=True)
                self._redis_client = None  # 将私有客户端属性置为空，后续再次获取时可再次触发实例化redis客户端逻辑
                break

        # 入队失败兜底：直接落库（应急方案）
        await fallback_save(msg, TempMessageStatus.REDIS_FAILED)
        return False

    async def get_message(self, timeout: int = 1, retry_times: int = 2) -> Optional[AgentMessage]:
        """
        消息出队（阻塞式，避免空轮询）
        :param timeout: 阻塞超时时间（秒），0=永久阻塞
        :param retry_times: 重试次数
        :return: AgentMessage/None
        """
        client = await self.get_redis_client()
        if client is None:
            logger.error("异步Redis客户端未初始化，无法从redis消息队列获取消息")
            return None
        # 重试逻辑：仅针对客户端超时异常
        for retry in range(retry_times + 1):
            try:
                # BRPOP：阻塞式出队，避免频繁查询空队列
                result = await client.brpop([self.queue_key], timeout=timeout)
                if not result:
                    return None
                # result格式：(队列key, 消息JSON字符串)
                msg_json = result[1]
                msg_dict = json.loads(msg_json)
                # 反序列化为AgentMessage
                msg = AgentMessage(**msg_dict)
                # 因为 brpop 是 “取出即删”，Redis 中无法保留消息的 “兜底副本”——临时表的 MQ_PENDING 状态，本质是 brpop 取出消息后的 “唯一兜底副本”
                await fallback_save(msg, TempMessageStatus.MQ_PENDING)  # 关键兜底 TODO 这里的兜底逻辑待优化，从redis取出后，到兜底存入数据库结束时，一旦出现异常情况，会存在数据丢失的风险
                return msg
                # 区分异常类型：仅处理客户端超时/连接异常
            except async_redis.TimeoutError as e:
                if retry >= retry_times:
                    logger.error(f"Redis BRPOP超时（重试{retry_times}次后仍失败）：{e}", exc_info=True)
                    return None
                logger.warning(f"Redis BRPOP超时，第{retry + 1}次重试...：{e}")
                # 重试前重置客户端（仅超时异常时重置）
                self._redis_client = None
                client = await self.get_redis_client()
                if client is None:
                    logger.error("重试时Redis客户端初始化失败，终止重试")
                    return None

            except async_redis.ConnectionError as e:
                logger.error(f"Redis连接异常：{e}", exc_info=True)
                self._redis_client = None
                return None

            # 其他异常（如JSON解析、参数错误）：不重试，直接返回
            except Exception as e:
                logger.error(f"消息出队Redis失败（非超时/连接异常）：{e}", exc_info=True)
                return None

    async def batch_get_messages(self, batch_size: int = 10, timeout: int = 1) -> List[AgentMessage]:
        """批量出队（提升消费效率）"""
        messages = []
        for _ in range(batch_size):
            msg = await self.get_message(timeout=timeout)
            if not msg:
                break
            messages.append(msg)
        return messages

    # 补偿逻辑（仅扫描temp_message表，逻辑极简）
    async def compensate_from_temp_table(self):
        """从temp_message表补偿兜底消息"""
        async for db in get_db():
            # 仅筛选：待恢复 + 24小时内的兜底消息
            temp_msgs = await db.execute(select(TempMessage).filter_by(status=TempMessageStatus.REDIS_FAILED, ))
            temp_msgs = temp_msgs.scalars().all()
            # 重新入Redis队列
            for msg in temp_msgs:
                agent_msg = AgentMessage(
                    message_id=msg.message_id,
                    session_id=msg.session_id,
                    message_type=msg.message_type,
                    content=msg.content,
                    generate_time=msg.generate_time
                )
                await self.put_message(agent_msg)
                # 标记为已恢复
                msg.status = TempMessageStatus.REDIS_SUCCESS
                await db.commit()  # 提交，使得记录更新生效


# 初始化Redis队列（全局单例）
redis_queue = RedisMessageQueue(
    redis_host="127.0.0.1",
    redis_port=6379,
    redis_db=0,
    queue_key="ops_qa_message_queue"
)


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
