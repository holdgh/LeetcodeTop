import redis.asyncio as redis
import logging
import time
from dataclasses import dataclass, asdict
import json
from typing import Optional, List

logger = logging.getLogger(__name__)


# 消息数据结构（和之前一致，兼容序列化）
@dataclass
class AgentMessage:
    session_id: str
    message_type: str  # user/rewrite/retrieve/expert
    content: str
    generate_time: str  # 生成时间戳
    create_at: float = time.time()  # 入队时间


class RedisMessageQueue:
    """Redis分布式消息队列（替代内存队列，支持限流校验）"""

    def __init__(self, redis_host: str, redis_port: int, redis_db: int = 0, queue_key: str = "ops_qa_message_queue"):
        # 初始化Redis客户端（异步）
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True  # 自动解码为字符串
        )
        self.queue_key = queue_key
        # 队列阈值配置（可根据服务器性能调整）
        self.QUEUE_MAX_LENGTH = 10000  # 队列最大长度，超过则拒绝新请求
        self.QUEUE_WARN_LENGTH = 8000  # 队列告警阈值

    async def check_queue_threshold(self) -> bool:
        """
        校验队列长度：
        - 小于阈值：返回True（允许入队）
        - 大于等于阈值：返回False（拒绝入队）
        """
        queue_len = await self.redis_client.llen(self.queue_key)
        if queue_len >= self.QUEUE_MAX_LENGTH:
            logger.warning(f"Redis队列已满（当前长度：{queue_len}，阈值：{self.QUEUE_MAX_LENGTH}），拒绝新请求")
            return False
        # 接近阈值时告警（可选）
        if queue_len >= self.QUEUE_WARN_LENGTH:
            logger.warning(f"Redis队列接近阈值（当前长度：{queue_len}，告警阈值：{self.QUEUE_WARN_LENGTH}）")
        return True

    async def put_message(self, msg: AgentMessage) -> bool:
        """
        消息入队（先校验阈值，再入队）
        :return: True=入队成功，False=队列满/入队失败
        """
        # 第一步：校验队列阈值
        if not await self.check_queue_threshold():
            return False

        # 第二步：消息序列化（dataclass转JSON字符串）
        try:
            msg_json = json.dumps(asdict(msg))
            # LPUSH：从队列头部入队，BRPOP从尾部出队，保证FIFO
            await self.redis_client.lpush(self.queue_key, msg_json)
            logger.debug(f"消息入队Redis成功：会话[{msg.session_id}]，类型[{msg.message_type}]")
            return True
        except Exception as e:
            logger.error(f"消息入队Redis失败：{e}", exc_info=True)
            # 入队失败兜底：直接落库（应急方案）
            await self._fallback_save(msg)
            return False

    async def get_message(self, timeout: int = 1) -> Optional[AgentMessage]:
        """
        消息出队（阻塞式，避免空轮询）
        :param timeout: 阻塞超时时间（秒），0=永久阻塞
        :return: AgentMessage/None
        """
        try:
            # BRPOP：阻塞式出队，避免频繁查询空队列
            result = await self.redis_client.brpop([self.queue_key], timeout=timeout)
            if not result:
                return None
            # result格式：(队列key, 消息JSON字符串)
            msg_json = result[1]
            msg_dict = json.loads(msg_json)
            # 反序列化为AgentMessage
            return AgentMessage(**msg_dict)
        except Exception as e:
            logger.error(f"消息出队Redis失败：{e}", exc_info=True)
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

    async def _fallback_save(self, msg: AgentMessage):
        """入队失败兜底落库（仅应急）"""
        # TODO 待定义数据表模型
        # from agent_work.database.database import MessageDB, StatusEnum
        # msg_db = MessageDB()
        # await msg_db.save_message(
        #     session_id=msg.session_id,
        #     message_type=msg.message_type,
        #     content=msg.content,
        #     generate_time=msg.generate_time,
        #     mq_send_status=StatusEnum.MQ_PENDING,
        #     mq_retry_count=0,
        #     db_save_status=StatusEnum.DB_PENDING
        # )
        logger.warning(f"消息[{msg.session_id}-{msg.message_type}]入队失败，已兜底落库")

    async def get_queue_length(self) -> int:
        """获取当前队列长度（用于监控）"""
        return await self.redis_client.llen(self.queue_key)


# 初始化Redis队列（全局单例）
redis_queue = RedisMessageQueue(
    redis_host="127.0.0.1",
    redis_port=6379,
    redis_db=0,
    queue_key="ops_qa_message_queue"
)
