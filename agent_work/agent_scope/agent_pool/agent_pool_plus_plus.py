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
            api_key="sk-6b8afa231399490bb7a56c025a3bc633",
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

    # async def bind_session(self, session_id: str) -> None:
    #     """绑定会话：记录会话信息"""
    #     # TODO 风险点1
    #     """
    #     bind_session仅记录session_id，未对实例加 “独占锁”—— 在极端高并发场景下，可能出现 “多个协程同时获取同一空闲实例” 的问题：
    #         协程 A 查询到实例 1 空闲，进入bind_session前被挂起；
    #         协程 B 同时查询到实例 1 仍标记为 “空闲”，也进入bind_session；
    #         最终实例 1 被两个会话绑定，导致数据污染（不同会话的memory混用）。
    #     """
    #     # # 绑定新会话
    #     # self.session_id = session_id
    #     # logger.info(f"Agent实例对 {self.pair_id} 绑定会话 {session_id}")
    #     # 新代码如下：
    #     async with self.instance_lock:  # 绑定前加锁，确保独占
    #         if self.session_id is not None:
    #             raise ValueError(f"实例 {self.pair_id} 已绑定会话 {self.session_id}")
    #         self.session_id = session_id
    #         logger.info(f"Agent实例对 {self.pair_id} 绑定会话 {session_id}")

    async def unbind_session(self) -> None:
        """解除会话绑定（含空闲通知）"""
        async with self.instance_lock:
            await self.retriever.memory.clear()
            await self.expert.memory.clear()
            await self.rewriter.memory.clear()
            # if self.session_id in SESSION_GOING_CACHE:
            #     del SESSION_GOING_CACHE[self.session_id]
            self.session_id = None
            logger.info(f"Agent实例对 {self.pair_id} 解除会话绑定，回归空闲")

        # 触发空闲通知（需将agent_pool设为全局/类属性，或通过参数传入）
        global agent_pool
        await agent_pool.notify_idle()


# -------------------------- Agent实例池 --------------------------
def _clean_agent_pair_resources(agent_pair: AgentPair):
    """清理AgentPair的资源（避免泄漏）"""
    # 主动置空实例的核心属性，断除内部引用链
    agent_pair.retriever = None
    agent_pair.expert = None
    agent_pair.rewriter = None
    agent_pair.model = None
    agent_pair.session_id = None


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
        # 1. 查找空闲实例
        idle_pairs = [p for p in self.pool if p.get_state() == AgentInstanceState.IDLE]
        if idle_pairs:
            for agent_pair in idle_pairs:
                try:
                    # 步骤1：快照校验：确认实例当前是否仍为空闲（避免脏读）
                    async with agent_pair.instance_lock:
                        if agent_pair.get_state() == AgentInstanceState.IDLE:
                            # 步骤2：尝试绑定会话（可能触发“已绑定”异常）
                            # await agent_pair.bind_session(session_id)
                            agent_pair.session_id = session_id
                            # 绑定成功 → 返回实例
                            return agent_pair
                except ValueError as e:
                    # 捕获“实例已被绑定”的异常（竞争失败），继续尝试下一个空闲实例
                    logger.warning(f"实例 {agent_pair.pair_id} 竞争失败（{str(e)}），尝试下一个实例")
                    continue  # 跳过当前实例，遍历下一个

        # 2. 无空闲实例，尝试扩容
        async with self.lock:
            if len(self.pool) < self.max_size:
                new_pair = AgentPair(pair_id=f"agent_pair_{len(self.pool) + 1}")
                # 这里是先绑定会话,然后再加入智能体实例池,属于同步操作,无需加协程锁校验
                # await new_pair.bind_session(session_id)
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
                idle_pairs = [p for p in self.pool if p.get_state() == AgentInstanceState.IDLE]
                if idle_pairs:
                    for agent_pair in idle_pairs:
                        try:
                            # 步骤1：快照校验：确认实例当前是否仍为空闲（避免脏读）
                            async with agent_pair.instance_lock:
                                if agent_pair.get_state() == AgentInstanceState.IDLE:
                                    # 步骤2：尝试绑定会话（可能触发“已绑定”异常）
                                    # await agent_pair.bind_session(session_id)
                                    agent_pair.session_id = session_id
                                    # 绑定成功 → 返回实例
                                    return agent_pair
                        except ValueError as e:
                            # 捕获“实例已被绑定”的异常（竞争失败），继续尝试下一个空闲实例
                            logger.warning(f"实例 {agent_pair.pair_id} 竞争失败（{str(e)}），尝试下一个实例")
                            continue  # 跳过当前实例，遍历下一个

                # 无空闲，挂起协程等待通知（最多等5秒，避免永久阻塞）
                try:
                    async with self.idle_condition:
                        await asyncio.wait_for(self.idle_condition.wait(), timeout=5)
                except asyncio.TimeoutError:
                    # 5秒无通知，自动唤醒重新检查（兜底）
                    continue

    # async def clean_expired_pairs(self):
    #     """定期清理过期实例（缩容加实例锁）"""
    #     try:
    #         while True:
    #             await asyncio.sleep(30)
    #             async with self.lock:
    #                 # 筛选真·空闲实例
    #                 real_idle_pairs = []
    #                 for p in self.pool:
    #                     async with p.instance_lock:
    #                         if p.get_state() == AgentInstanceState.IDLE:
    #                             real_idle_pairs.append(p)
    #
    #                 # 缩容逻辑
    #                 idle_count = len(real_idle_pairs)
    #                 if idle_count > self.min_size and len(self.pool) > self.min_size:
    #                     redundant_pairs = real_idle_pairs[self.min_size:]
    #                     for p in redundant_pairs:
    #                         self.pool.remove(p)
    #                     logger.info(
    #                         f"实例池缩容，当前容量：{len(self.pool)}，空闲数：{len(real_idle_pairs) - len(redundant_pairs)}")
    #     except asyncio.CancelledError:
    #         logger.info("清理过期实例的后台任务已被取消")
    #         return

    async def clean_expired_pairs(self):
        """定期清理过期实例（优化：锁粒度最小化，避免阻塞整个实例池）"""
        try:
            while True:
                await asyncio.sleep(30)
                logger.info(f"开始清理过期实例，当前池容量：{len(self.pool)}")

                # ========== 步骤1：无锁快照筛选疑似空闲实例（无池级锁，不阻塞任何操作） ==========
                idle_pairs = [p for p in self.pool if p.get_state() == AgentInstanceState.IDLE]

                if not idle_pairs:
                    logger.info("无疑似空闲实例，跳过清理")
                    continue

                # ========== 步骤2：加实例锁确认真·空闲实例（无池级锁） ==========
                real_idle_pairs = []
                for p in idle_pairs:
                    try:
                        async with p.instance_lock:
                            # 锁内最终确认是否空闲
                            if p.get_state() == AgentInstanceState.IDLE:
                                real_idle_pairs.append(p)
                    except Exception as e:
                        logger.warning(f"校验实例 {p.pair_id} 状态失败：{str(e)}")
                        continue

                # ========== 步骤3：仅在删除实例时加池级锁（最小化持有时间） ==========
                idle_count = len(real_idle_pairs)
                # 缩容阈值：仅当空闲实例数 > min_size 且 池总容量 > min_size 时缩容
                if idle_count > self.min_size and len(self.pool) > self.min_size:
                    # 计算需要删除的实例数：保留min_size个空闲实例
                    need_delete_count = idle_count - self.min_size
                    # 取前need_delete_count个空闲实例（避免删除所有）
                    redundant_pairs = real_idle_pairs[:need_delete_count]

                    # 加池级锁删除实例（仅这一步持有池级锁，耗时极短）
                    async with self.lock:
                        deleted_count = 0
                        for p in redundant_pairs:
                            if p in self.pool:  # 二次校验，避免重复删除
                                # 清理实例资源（核心：避免内存/连接泄漏）
                                _clean_agent_pair_resources(p)
                                self.pool.remove(p)
                                deleted_count += 1

                    logger.info(
                        f"实例池缩容完成，删除 {deleted_count} 个空闲实例，当前容量：{len(self.pool)}，剩余空闲数：{idle_count - deleted_count}")
                else:
                    logger.info(f"无需缩容，当前空闲实例数：{idle_count}，最小空闲数：{self.min_size}")

        except asyncio.CancelledError:
            logger.info("清理过期实例的后台任务已被取消")
            return
        except Exception as e:
            logger.error(f"清理过期实例时发生异常：{str(e)}", exc_info=True)

    async def notify_idle(self):
        """触发空闲实例通知"""
        async with self.idle_condition:
            self.idle_condition.notify_all()


# 全局实例池（供unbind_session调用）
agent_pool = AgentPool(min_size=5, max_size=20)
