from datetime import datetime, timedelta
from enum import Enum
import asyncio
from typing import List, Optional, Callable, TypeVar, Generic, Tuple
from agentscope.agent import ReActAgent
from agentscope.model import DashScopeChatModel
import logging
from cachetools import TTLCache
from weakref import WeakKeyDictionary  # 弱引用字典，避免内存泄漏

# 日志配置（复用你的原有配置）
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 泛型类型定义：支持任意 ReActAgent 子类
AgentType = TypeVar("AgentType", bound=ReActAgent)


# -------------------------- 智能体状态枚举（复用，无需修改） --------------------------
class AgentInstanceState(Enum):
    IDLE = "空闲"  # 可被复用
    BUSY = "忙碌"  # 已绑定会话，正在处理


# -------------------------- 泛型智能体池（核心类） --------------------------
class GenericAgentPool(Generic[AgentType]):
    """
    泛型智能体池：支持按智能体类型独立管理（重写、检索、专家分别建池）
    特性：动态扩缩容、并发安全、闲置时间判断、泛型适配
    """

    def __init__(
            self,
            agent_creator: Callable[[], AgentType],  # 智能体创建函数（如 create_rewrite_agent）
            agent_type_name: str,  # 智能体类型名称（用于日志区分，如 "重写助手"）
            min_size: int = 3,  # 最小实例数（预创建）
            max_size: int = 20,  # 最大实例数（扩容上限）
            idle_timeout: int = 30,  # 实例闲置超时时间（秒，用于缩容）
            wait_timeout: int = 120  # 获取实例超时时间（秒）
    ):
        self.agent_creator = agent_creator  # 智能体创建工厂函数
        self.agent_type_name = agent_type_name  # 智能体类型标识
        self.min_size = min_size
        self.max_size = max_size
        self.idle_timeout = idle_timeout
        self.wait_timeout = wait_timeout

        # 核心存储：智能体实例列表
        self.pool: List[AgentType] = []
        # 弱引用字典：存储每个智能体的元数据（避免强引用导致内存泄漏）
        # key: Agent实例，value: {"session_id": 绑定的会话ID, "lock": 实例级锁, "last_release_time": 最后释放时间}
        self.agent_meta: WeakKeyDictionary[AgentType, dict] = WeakKeyDictionary()

        # 并发安全锁
        self.pool_lock = asyncio.Lock()  # 池级锁（操作实例列表时使用）
        self.idle_event = asyncio.Event()  # 空闲实例事件（替代轮询，提升等待效率）

    async def init_pool(self) -> None:
        """初始化实例池：预创建 min_size 个智能体实例（服务启动时调用）"""
        async with self.pool_lock:
            for i in range(self.min_size):
                agent = self._create_agent()
                self.pool.append(agent)
                # 初始化元数据
                self.agent_meta[agent] = {
                    "session_id": None,
                    "lock": asyncio.Lock(),
                    "last_release_time": datetime.now()
                }
            logger.info(
                f"【{self.agent_type_name}实例池】初始化完成，预创建 {self.min_size} 个实例，当前容量：{len(self.pool)}"
            )

    def _create_agent(self) -> AgentType:
        """创建单个智能体实例（内部调用，封装创建逻辑）"""
        try:
            agent = self.agent_creator()
            # 为智能体添加实例ID（便于日志追踪）
            agent.instance_id = f"{self.agent_type_name}_instance_{len(self.pool) + 1}"
            return agent
        except Exception as e:
            logger.error(f"【{self.agent_type_name}实例池】创建实例失败：{str(e)}", exc_info=True)
            raise

    async def get_agent(self, session_id: str) -> Tuple[AgentType, str]:
        """
        获取智能体实例（核心接口）
        Args:
            session_id: 会话ID
        Returns:
            Tuple[AgentType, str]: (智能体实例, 实例ID)
        Raises:
            TimeoutError: 获取实例超时
        """
        # 第一步：尝试从空闲实例中获取
        async with self.pool_lock:
            idle_agents = self._get_idle_agents()
            if idle_agents:
                for agent in idle_agents:
                    if await self._try_bind_agent(agent, session_id):
                        logger.info(
                            f"【{self.agent_type_name}实例池】会话 {session_id} 复用空闲实例 {agent.instance_id}"
                        )
                        return agent, agent.instance_id

        # 第二步：无空闲实例，尝试扩容
        if await self._try_expand_pool():
            # 扩容后递归获取（新实例已绑定会话）
            return await self.get_agent(session_id)

        # 第三步：已达最大容量，等待空闲实例
        logger.warning(
            f"【{self.agent_type_name}实例池】已达最大容量 {self.max_size}，会话 {session_id} 进入等待队列"
        )
        try:
            agent = await self._wait_for_idle_agent(session_id)
            return agent, agent.instance_id
        except TimeoutError as e:
            logger.error(f"【{self.agent_type_name}实例池】会话 {session_id} 获取实例超时：{str(e)}")
            raise

    async def release_agent(self, agent: AgentType, session_id: str) -> None:
        """
        释放智能体实例（核心接口）：解绑会话+清理内存+触发空闲事件
        Args:
            agent: 要释放的智能体实例
            session_id: 会话ID（用于校验）
        """
        # 获取实例元数据（弱引用字典，避免实例已被销毁的情况）
        meta = self.agent_meta.get(agent)
        if not meta:
            logger.warning(f"【{self.agent_type_name}实例池】释放失败：实例 {agent.instance_id} 元数据不存在")
            return

        async with meta["lock"]:
            # 校验实例是否属于当前会话（避免释放其他会话的实例）
            if meta["session_id"] != session_id:
                logger.warning(
                    f"【{self.agent_type_name}实例池】释放失败：实例 {agent.instance_id} 未绑定会话 {session_id}"
                )
                return

            # 清理智能体内存（关键：避免会话数据污染）
            await agent.memory.clear()
            # 解绑会话+更新释放时间
            meta["session_id"] = None
            meta["last_release_time"] = datetime.now()

        logger.info(f"【{self.agent_type_name}实例池】实例 {agent.instance_id} 释放成功，回归空闲")
        # 触发空闲事件（通知等待队列）
        self.idle_event.set()

    async def start_cleanup_task(self) -> None:
        """启动后台清理任务（缩容+剔除长期闲置实例）"""
        try:
            while True:
                await asyncio.sleep(30)  # 每30秒检查一次
                await self._cleanup_idle_agents()
        except asyncio.CancelledError:
            logger.info(f"【{self.agent_type_name}实例池】后台清理任务已取消（服务关闭/重载）")
            return

    # -------------------------- 内部辅助方法 --------------------------
    def _get_idle_agents(self) -> List[AgentType]:
        """获取所有空闲实例（内部调用，需在池级锁保护下执行）"""
        idle_agents = []
        for agent in self.pool:
            meta = self.agent_meta.get(agent)
            if meta and meta["session_id"] is None:
                idle_agents.append(agent)
        return idle_agents

    async def _try_bind_agent(self, agent: AgentType, session_id: str) -> bool:
        """尝试绑定智能体到会话（内部调用，处理并发竞争）"""
        meta = self.agent_meta.get(agent)
        if not meta:
            return False

        # 实例级锁：确保同一时间只有一个会话能绑定该实例
        async with meta["lock"]:
            # 二次校验：避免其他协程已绑定该实例（脏读防护）
            if meta["session_id"] is not None:
                return False
            # 绑定会话
            meta["session_id"] = session_id
            return True

    async def _try_expand_pool(self) -> bool:
        """尝试扩容实例池（内部调用）"""
        async with self.pool_lock:
            current_size = len(self.pool)
            if current_size >= self.max_size:
                return False  # 已达最大容量，无法扩容

            # 创建新实例并绑定会话
            new_agent = self._create_agent()
            self.pool.append(new_agent)
            # 初始化元数据并直接绑定会话
            self.agent_meta[new_agent] = {
                "session_id": None,  # 后续由 get_agent 绑定
                "lock": asyncio.Lock(),
                "last_release_time": datetime.now()
            }
            logger.info(
                f"【{self.agent_type_name}实例池】扩容成功，当前容量：{len(self.pool)}/{self.max_size}"
            )
            return True

    async def _wait_for_idle_agent(self, session_id: str) -> AgentType:
        """等待空闲实例（内部调用，事件驱动）"""
        start_time = datetime.now()
        while datetime.now() - start_time < timedelta(seconds=self.wait_timeout):
            # 等待空闲事件触发（最多等待2秒，避免无限阻塞）
            try:
                await asyncio.wait_for(self.idle_event.wait(), timeout=2)
                self.idle_event.clear()  # 重置事件
            except asyncio.TimeoutError:
                pass  # 超时继续循环，检查是否超时

            # 检查是否有空闲实例
            async with self.pool_lock:
                idle_agents = self._get_idle_agents()
                for agent in idle_agents:
                    if await self._try_bind_agent(agent, session_id):
                        logger.info(
                            f"【{self.agent_type_name}实例池】会话 {session_id} 等待成功，获取实例 {agent.instance_id}"
                        )
                        return agent

        # 超时抛出异常
        raise TimeoutError(f"等待 {self.wait_timeout} 秒后仍无空闲实例，请稍后重试")

    async def _cleanup_idle_agents(self) -> None:
        """清理长期闲置的实例（缩容逻辑，内部调用）"""
        async with self.pool_lock:
            current_size = len(self.pool)
            if current_size <= self.min_size:
                return  # 已达最小容量，不缩容

            # 筛选：空闲且闲置时间超过阈值的实例
            idle_agents = self._get_idle_agents()
            if len(idle_agents) <= self.min_size:
                return  # 空闲实例不足，不缩容

            # 按闲置时间排序（先销毁闲置最久的）
            idle_agents.sort(
                key=lambda x: self.agent_meta[x]["last_release_time"],
                reverse=False
            )

            # 计算需要销毁的实例数（保留 min_size 个空闲实例）
            need_destroy_count = len(idle_agents) - self.min_size
            if need_destroy_count <= 0:
                return

            # 销毁实例
            destroyed_agents = idle_agents[:need_destroy_count]
            for agent in destroyed_agents:
                self.pool.remove(agent)
                del self.agent_meta[agent]  # 移除元数据
                # 可选：销毁模型连接（如果智能体持有独立模型）
                agent.model = None
                logger.info(
                    f"【{self.agent_type_name}实例池】缩容销毁实例 {agent.instance_id}，当前容量：{len(self.pool)}"
                )


# -------------------------- 实例池初始化（按智能体类型独立创建） --------------------------
# 假设你已定义以下智能体创建函数（复用你的原有逻辑）
from agent_work.agent_scope.agent.search_agent import create_retriever_agent
from agent_work.agent_scope.agent.expert_agent import create_expert_agent
from agent_work.agent_scope.agent.rewrite_agent import create_rewrite_agent


# 1. 全局模型池（复用之前的优化方案，所有智能体共享模型，减少内存占用）
class LLMModelPool:
    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def init(self):
        """初始化全局共享模型（服务启动时调用）"""
        async with self._lock:
            if hasattr(self, "model"):
                return
            self.model = DashScopeChatModel(
                model_name="deepseek-v3",
                api_key="sk-f61034a0afd64ffdab4be83a063b20e3",
                generate_kwargs={
                    "temperature": 0.1,
                    "top_p": 0.8,
                    "max_tokens": 300,
                    "repetition_penalty": 1.1
                }
            )


# 初始化全局模型池（服务启动时执行）
llm_pool = LLMModelPool()


# 2. 重写智能体创建函数（适配共享模型）
async def create_rewrite_agent_with_shared_model() -> ReActAgent:
    """重写智能体创建函数（使用全局共享模型）"""
    await llm_pool.init()  # 确保模型已初始化
    return create_rewrite_agent(llm_pool.model)  # 传入共享模型


# 3. 检索智能体创建函数（适配共享模型）
async def create_retriever_agent_with_shared_model() -> ReActAgent:
    await llm_pool.init()
    return create_retriever_agent(llm_pool.model)


# 4. 专家智能体创建函数（适配共享模型）
async def create_expert_agent_with_shared_model() -> ReActAgent:
    await llm_pool.init()
    return create_expert_agent(llm_pool.model)


# 5. 创建3个独立的实例池（按智能体类型拆分）
async def init_all_agent_pools() -> Tuple[GenericAgentPool, GenericAgentPool, GenericAgentPool]:
    """初始化所有智能体池（服务启动时调用）"""
    # 重写助手池：耗时短、可多实例
    rewrite_pool = GenericAgentPool(
        agent_creator=create_rewrite_agent_with_shared_model,
        agent_type_name="重写助手",
        min_size=5,
        max_size=30,
        idle_timeout=30,
        wait_timeout=120
    )

    # 检索助手池：耗时短、可多实例
    retriever_pool = GenericAgentPool(
        agent_creator=create_retriever_agent_with_shared_model,
        agent_type_name="检索助手",
        min_size=5,
        max_size=30,
        idle_timeout=30,
        wait_timeout=120
    )

    # 专家助手池：耗时长、资源占用高，限制最大实例数
    expert_pool = GenericAgentPool(
        agent_creator=create_expert_agent_with_shared_model,
        agent_type_name="运维专家",
        min_size=2,
        max_size=15,
        idle_timeout=30,
        wait_timeout=120
    )

    # 初始化池并启动清理任务
    await rewrite_pool.init_pool()
    await retriever_pool.init_pool()
    await expert_pool.init_pool()

    # 启动后台清理任务（缩容）
    asyncio.create_task(rewrite_pool.start_cleanup_task())
    asyncio.create_task(retriever_pool.start_cleanup_task())
    asyncio.create_task(expert_pool.start_cleanup_task())

    logger.info("所有智能体池初始化完成！")
    return rewrite_pool, retriever_pool, expert_pool


# -------------------------- 业务调用示例（如何使用独立实例池） --------------------------
async def process_user_request(session_id: str, user_question: str):
    """处理用户请求：从3个独立池获取智能体，协作完成回复"""
    # 1. 初始化实例池（实际项目中在服务启动时执行一次）
    rewrite_pool, retriever_pool, expert_pool = await init_all_agent_pools()

    # 2. 获取重写助手实例
    rewrite_agent, rewrite_instance_id = await rewrite_pool.get_agent(session_id)
    try:
        # 调用重写助手（假设返回 Msg 实例）
        rewritten_msg = await rewrite_agent.async_run(user_question)
    finally:
        # 释放重写助手（用完即释放，其他会话可复用）
        await rewrite_pool.release_agent(rewrite_agent, session_id)

    # 3. 获取检索助手实例
    retriever_agent, retriever_instance_id = await retriever_pool.get_agent(session_id)
    try:
        # 调用检索助手
        retrieval_msg = await retriever_agent.async_run(rewritten_msg)
    finally:
        await retriever_pool.release_agent(retriever_agent, session_id)

    # 4. 获取专家助手实例
    expert_agent, expert_instance_id = await expert_pool.get_agent(session_id)
    try:
        # 调用专家助手
        expert_response = await expert_agent.async_run(rewritten_msg, retrieval_msg)
    finally:
        await expert_pool.release_agent(expert_agent, session_id)

    return expert_response