from datetime import datetime, timedelta
from enum import Enum
import asyncio
from typing import List, Optional
from agentscope.agent import ReActAgent
from agentscope.model import DashScopeChatModel
import logging
from cachetools import TTLCache

from agent_work.agent_scope.agent.expert_agent import create_expert_agent
from agent_work.agent_scope.agent.rewrite_agent import create_rewrite_agent
from agent_work.agent_scope.agent.search_agent import create_retriever_agent

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# 缓存配置：key=会话id，value=当前会话是否进行中的标识
SESSION_GOING_CACHE = TTLCache(maxsize=1000, ttl=1800)


# -------------------------- Agent实例状态枚举 --------------------------
class AgentInstanceState(Enum):
    IDLE = "空闲"
    BUSY = "忙碌"


# -------------------------- 单个Agent实例封装 --------------------------
class AgentPair:
    def __init__(self, pair_id: str):
        self.pair_id = pair_id
        self.session_id: Optional[str] = None
        self.message_id: Optional[str] = None
        self.model = DashScopeChatModel(
            model_name="qwen3-max",
            api_key="sk-******",
            generate_kwargs={
                "temperature": 0.1,
                "top_p": 0.8,
                "max_tokens": 300,
                "repetition_penalty": 1.1
            }
        )
        self.instance_lock = asyncio.Lock()
        self.retriever = self._create_retriever_agent()
        self.expert = self._create_expert_agent()
        self.rewriter = self._create_rewrite_agent()

    def _create_retriever_agent(self) -> ReActAgent:
        return create_retriever_agent(self.model)

    def _create_expert_agent(self) -> ReActAgent:
        return create_expert_agent(self.model)

    def _create_rewrite_agent(self) -> ReActAgent:
        return create_rewrite_agent(self.model)

    def get_state(self) -> AgentInstanceState:
        if self.session_id is None:
            return AgentInstanceState.IDLE
        return AgentInstanceState.BUSY

    async def unbind_session(self) -> None:
        """解除会话绑定（含空闲通知）"""
        async with self.instance_lock:
            await self.retriever.memory.clear()
            await self.expert.memory.clear()
            await self.rewriter.memory.clear()
            if self.session_id in SESSION_GOING_CACHE:
                del SESSION_GOING_CACHE[self.session_id]
            self.session_id = None
            logger.info(f"Agent实例对 {self.pair_id} 解除会话绑定，回归空闲")

        # 触发空闲通知（需将agent_pool设为全局/类属性，或通过参数传入）
        global agent_pool
        await agent_pool.notify_idle()


# -------------------------- Agent实例池 --------------------------
class AgentPool:
    def __init__(self, min_size: int = 5, max_size: int = 20):
        self.min_size = min_size
        self.max_size = max_size
        self.pool: List[AgentPair] = []
        self.lock = asyncio.Lock()
        self.idle_condition = asyncio.Condition(self.lock)  # 空闲通知条件变量

    async def init_pool(self):
        """初始化实例池"""
        async with self.lock:
            for i in range(self.min_size):
                agent_pair = AgentPair(pair_id=f"agent_pair_{i + 1}")
                self.pool.append(agent_pair)
            logger.info(f"实例池初始化完成，预创建 {self.min_size} 个Agent实例对")

    async def get_agent_pair(self, session_id: str) -> AgentPair:
        """获取会话对应的Agent实例对（原子操作避免并发抢占）"""
        # 1. 遍历现有实例，尝试原子绑定
        for agent_pair in self.pool:
            try:
                async with agent_pair.instance_lock:
                    if agent_pair.get_state() == AgentInstanceState.IDLE:
                        agent_pair.session_id = session_id
                        logger.info(f"Agent实例对 {agent_pair.pair_id} 绑定会话 {session_id}")
                        return agent_pair
            except Exception as e:
                logger.warning(f"尝试绑定实例 {agent_pair.pair_id} 失败：{str(e)}")
                continue

        # 2. 无空闲实例，尝试扩容
        async with self.lock:
            if len(self.pool) < self.max_size:
                new_pair = AgentPair(pair_id=f"agent_pair_{len(self.pool) + 1}")
                async with new_pair.instance_lock:
                    new_pair.session_id = session_id
                self.pool.append(new_pair)
                logger.info(f"实例池扩容，当前容量：{len(self.pool)}")
                return new_pair

        # 3. 已达最大容量，等待空闲实例
        try:
            return await self._wait_for_idle_pair(session_id, timeout=120)
        except TimeoutError as e:
            raise TimeoutError(f"获取空闲实例超时：{str(e)}") from e

    async def _wait_for_idle_pair(self, session_id: str, timeout: int = 60) -> AgentPair:
        """等待空闲实例（条件变量替代轮询）"""
        start_time = datetime.now()
        async with self.idle_condition:
            while True:
                if datetime.now() - start_time > timedelta(seconds=timeout):
                    raise TimeoutError("当前咨询用户过多，请稍后重试")

                # 尝试绑定空闲实例
                for agent_pair in self.pool:
                    try:
                        async with agent_pair.instance_lock:
                            if agent_pair.get_state() == AgentInstanceState.IDLE:
                                agent_pair.session_id = session_id
                                logger.info(f"会话 {session_id} 等待后获取实例 {agent_pair.pair_id}")
                                return agent_pair
                    except Exception as e:
                        logger.warning(f"等待时绑定实例 {agent_pair.pair_id} 失败：{str(e)}")
                        continue

                # 无空闲，等待通知
                await asyncio.wait_for(self.idle_condition.wait(), timeout=5)

    async def clean_expired_pairs(self):
        """定期清理过期实例（缩容加实例锁）"""
        try:
            while True:
                await asyncio.sleep(30)
                async with self.lock:
                    # 筛选真·空闲实例
                    real_idle_pairs = []
                    for p in self.pool:
                        async with p.instance_lock:
                            if p.get_state() == AgentInstanceState.IDLE:
                                real_idle_pairs.append(p)

                    # 缩容逻辑
                    idle_count = len(real_idle_pairs)
                    if idle_count > self.min_size and len(self.pool) > self.min_size:
                        redundant_pairs = real_idle_pairs[self.min_size:]
                        for p in redundant_pairs:
                            self.pool.remove(p)
                        logger.info(
                            f"实例池缩容，当前容量：{len(self.pool)}，空闲数：{len(real_idle_pairs) - len(redundant_pairs)}")
        except asyncio.CancelledError:
            logger.info("清理过期实例的后台任务已被取消")
            return

    async def notify_idle(self):
        """触发空闲实例通知"""
        async with self.idle_condition:
            self.idle_condition.notify_all()


# 全局实例池（供unbind_session调用）
agent_pool = AgentPool(min_size=5, max_size=20)