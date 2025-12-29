from datetime import datetime, timedelta
from enum import Enum
import asyncio
from typing import List
from agentscope.agent import ReActAgent
from agentscope.model import DashScopeChatModel
import logging
from cachetools import TTLCache  # 带过期时间的缓存，避免内存泄漏
from agent_work.agent_scope.agent.search_agent import create_retriever_agent
from agent_work.agent_scope.agent.expert_agent import create_expert_agent
from agent_work.agent_scope.agent.rewrite_agent import create_rewrite_agent


# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# 缓存配置：key=会话id，value=当前会话是否进行中的标识，0标识空闲，1表示进行中，30分钟误操作自动过期
SESSION_GOING_CACHE = TTLCache(maxsize=1000, ttl=1800)  # 最多缓存1000个会话，30分钟无操作自动过期
from typing import Optional


# -------------------------- Agent实例状态枚举 --------------------------
class AgentInstanceState(Enum):
    IDLE = "空闲"  # 可被复用
    BUSY = "忙碌"  # 已绑定会话，正在处理


# -------------------------- 单个Agent实例封装（含检索+专家对） --------------------------
class AgentPair:
    """Agent实例对：包含一个检索Agent和一个专家Agent，绑定同一会话"""

    def __init__(self, pair_id: str):
        self.pair_id = pair_id  # 实例对ID
        self.session_id: Optional[str] = None  # 绑定的会话ID（空闲时为None）
        self.message_id: Optional[str] = None  # 绑定的对话消息ID（空闲时为None）
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

        self.instance_lock = asyncio.Lock()  # 实例级锁
        # 创建检索Agent和专家Agent
        self.retriever = self._create_retriever_agent()
        self.expert = self._create_expert_agent()
        self.rewriter = self._create_rewrite_agent()

    def _create_retriever_agent(self) -> ReActAgent:
        return create_retriever_agent(self.model)

    def _create_expert_agent(self) -> ReActAgent:
        return create_expert_agent(self.model)

    def _create_rewrite_agent(self) -> ReActAgent:
        return create_rewrite_agent(self.model)

    async def bind_session(self, session_id: str) -> None:
        """绑定会话：清理内存+记录会话信息"""
        # 清理上一会话的内存（核心：保证隔离）TODO 由于解绑时已经清理智能体实例缓存，此处可不再清理
        # await self.retriever.memory.clear()
        # await self.expert.memory.clear()
        # TODO 风险点1
        """
        bind_session仅记录session_id，未对实例加 “独占锁”—— 在极端高并发场景下，可能出现 “多个协程同时获取同一空闲实例” 的问题：
            协程 A 查询到实例 1 空闲，进入bind_session前被挂起；
            协程 B 同时查询到实例 1 仍标记为 “空闲”，也进入bind_session；
            最终实例 1 被两个会话绑定，导致数据污染（不同会话的memory混用）。
        """
        # # 绑定新会话
        # self.session_id = session_id
        # logger.info(f"Agent实例对 {self.pair_id} 绑定会话 {session_id}")
        # 新代码如下：
        async with self.instance_lock:  # 绑定前加锁，确保独占
            if self.session_id is not None:
                raise ValueError(f"实例 {self.pair_id} 已绑定会话 {self.session_id}")
            self.session_id = session_id
            logger.info(f"Agent实例对 {self.pair_id} 绑定会话 {session_id}")

    async def unbind_session(self) -> None:
        """解除会话绑定：清理内存+恢复空闲状态"""
        # await self.retriever.memory.clear()
        # await self.expert.memory.clear()
        # # 将当前会话从会话是否进行中的缓存中清除
        # if self.session_id in SESSION_GOING_CACHE:
        #     del SESSION_GOING_CACHE[self.session_id]
        # self.session_id = None
        # logger.info(f"Agent实例对 {self.pair_id} 解除会话绑定，回归空闲")
        # 新代码如下：
        async with self.instance_lock:  # 解绑前加锁，避免并发修改
            await self.retriever.memory.clear()
            await self.expert.memory.clear()
            await self.rewriter.memory.clear()
            if self.session_id in SESSION_GOING_CACHE:
                del SESSION_GOING_CACHE[self.session_id]
            self.session_id = None
            logger.info(f"Agent实例对 {self.pair_id} 解除会话绑定，回归空闲")

    def get_state(self) -> AgentInstanceState:
        """获取当前实例状态"""
        if self.session_id is None:
            return AgentInstanceState.IDLE
        return AgentInstanceState.BUSY


# -------------------------- Agent实例池（核心：动态调度+复用） --------------------------
class AgentPool:
    def __init__(self, min_size: int = 5, max_size: int = 20):
        self.min_size = min_size  # 实例池最小容量（预创建）
        self.max_size = max_size  # 实例池最大容量（扩容上限）
        self.pool: List[AgentPair] = []  # 实例池存储
        self.lock = asyncio.Lock()  # 并发安全锁

    async def init_pool(self):
        """初始化实例池：预创建min_size个Agent实例对"""
        async with self.lock:
            for i in range(self.min_size):
                agent_pair = AgentPair(pair_id=f"agent_pair_{i + 1}")
                self.pool.append(agent_pair)
            logger.info(f"实例池初始化完成，预创建 {self.min_size} 个Agent实例对")

    async def get_agent_pair(self, session_id: str) -> AgentPair:
        """获取会话对应的Agent实例对（复用空闲实例或扩容）"""
        async with self.lock:
            # 2. 查找空闲实例
            idle_pairs = [p for p in self.pool if p.get_state() == AgentInstanceState.IDLE]
            if idle_pairs:
                for agent_pair in idle_pairs:
                    try:
                        # 步骤1：快照校验：确认实例当前是否仍为空闲（避免脏读）
                        async with agent_pair.instance_lock:
                            is_really_idle = (agent_pair.get_state() == AgentInstanceState.IDLE)
                        if not is_really_idle:
                            continue  # 实例已被其他协程抢占，跳过

                        # 步骤2：尝试绑定会话（可能触发“已绑定”异常）
                        await agent_pair.bind_session(session_id)
                        # 绑定成功 → 返回实例
                        return agent_pair

                    except ValueError as e:
                        # 捕获“实例已被绑定”的异常（竞争失败），继续尝试下一个空闲实例
                        logger.warning(f"实例 {agent_pair.pair_id} 竞争失败（{str(e)}），尝试下一个实例")
                        continue  # 跳过当前实例，遍历下一个

            # 3. 无空闲实例，且未达最大容量→扩容
            if len(self.pool) < self.max_size:
                new_pair = AgentPair(pair_id=f"agent_pair_{len(self.pool) + 1}")
                # 这里是先绑定会话,然后再加入智能体实例池,属于同步操作,无需加协程锁校验
                await new_pair.bind_session(session_id)
                self.pool.append(new_pair)
                logger.info(f"实例池扩容，当前容量：{len(self.pool)}")
                return new_pair

            # 4. 已达最大容量→等待空闲实例（超时2分钟）
            try:
                return await self._wait_for_idle_pair(session_id, timeout=120)  # 排队时长增加到2分钟
            except TimeoutError as e:
                raise TimeoutError(f"获取空闲实例超时：{str(e)}") from e

    async def _wait_for_idle_pair(self, session_id: str, timeout: int = 60) -> AgentPair:
        """等待空闲实例（超时抛出异常）"""
        start_time = datetime.now()
        while datetime.now() - start_time < timedelta(seconds=timeout):
            await asyncio.sleep(5)  # 每5秒检查一次
            async with self.lock:
                idle_pairs = [p for p in self.pool if p.get_state() == AgentInstanceState.IDLE]
                if idle_pairs:
                    for agent_pair in idle_pairs:
                        try:
                            # 快照校验 + 尝试绑定
                            async with agent_pair.instance_lock:
                                is_really_idle = (agent_pair.get_state() == AgentInstanceState.IDLE)
                            if not is_really_idle:
                                continue

                            await agent_pair.bind_session(session_id)
                            logger.info(f"会话 {session_id} 成功获取空闲实例 {agent_pair.pair_id}")
                            return agent_pair

                        except ValueError as e:
                            logger.warning(f"等待期间实例 {agent_pair.pair_id} 竞争失败：{str(e)}，继续等待")
                            continue  # 跳过已被抢占的实例，继续等待下一次检查
        raise TimeoutError("当前咨询用户过多，请稍后重试")

    async def clean_expired_pairs(self):
        """定期清理过期实例（解除绑定，回归空闲）"""
        try:
            while True:
                await asyncio.sleep(30)  # 每半分钟检查一次
                async with self.lock:

                    # 缩容：若空闲实例过多，且超过最小容量→销毁多余实例
                    idle_count = len([p for p in self.pool if p.get_state() == AgentInstanceState.IDLE])
                    if idle_count > self.min_size and len(self.pool) > self.min_size:
                        # 保留min_size个空闲实例，销毁其余
                        idle_pairs = [p for p in self.pool if p.get_state() == AgentInstanceState.IDLE]
                        redundant_pairs = idle_pairs[self.min_size:]
                        for p in redundant_pairs:
                            self.pool.remove(p)
                        logger.info(f"实例池缩容，当前容量：{len(self.pool)}")
        except asyncio.CancelledError:
            # 捕获任务取消异常，静默退出（无需报错）
            logger.info("清理过期实例的后台任务已被取消（服务重载/关闭）")
            return
