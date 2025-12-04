from datetime import datetime, timedelta
from enum import Enum
import asyncio
from typing import List, Optional, Callable, TypeVar, Generic, Tuple, Dict, Any
from agentscope.agent import ReActAgent
from agentscope.model import DashScopeChatModel, ChatModelBase
import logging
from weakref import WeakKeyDictionary  # 弱引用字典，避免内存泄漏
from agent_work.agent_scope.agent.search_agent import create_retriever_agent
from agent_work.agent_scope.agent.expert_agent import create_expert_agent
from agent_work.agent_scope.agent.rewrite_agent import create_rewrite_agent

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
            agent_creator: Callable[[ChatModelBase,  asyncio.Lock], AgentType],  # 智能体创建函数（如 create_rewrite_agent）
            agent_type_name: str,  # 智能体类型名称（用于日志区分，如 "重写助手"）
            # 模型池配置（每个智能体池独立配置）
            llm_model_name: str = "deepseek-v3",
            llm_api_key: str = "sk-6b8afa231399490bb7a56c025a3bc633",
            llm_model_count: int = 3,
            llm_generate_kwargs: Optional[Dict[str, Any]] = None,
            # 智能体池基础配置
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

        # 绑定专属模型池（核心：一个智能体池对应一个模型池）
        self.llm_pool = LLMModelPool()
        self.llm_model_name = llm_model_name
        self.llm_api_key = llm_api_key
        self.llm_model_count = llm_model_count
        self.llm_generate_kwargs = llm_generate_kwargs or {}

        # 核心存储：智能体实例列表
        self.pool: List[AgentType] = []
        # 弱引用字典：存储每个智能体的元数据（避免强引用导致内存泄漏）【当智能体实例被回收时，弱引用字典可自动删除相应的键值信息】
        # key: Agent实例，value: {"session_id": 绑定的会话ID, "lock": 实例级锁, "last_release_time": 最后释放时间}
        self.agent_meta: WeakKeyDictionary[AgentType, dict] = WeakKeyDictionary()

        # 并发安全锁
        self.pool_lock = asyncio.Lock()  # 池级锁（操作实例列表时使用）
        self.idle_event = asyncio.Event()  # 空闲实例事件（替代轮询，提升等待效率）

    async def init_pool(self) -> None:
        """初始化智能体池（先初始化专属模型池，再创建智能体）"""
        # 1. 先初始化专属模型池
        await self.llm_pool.init(
            model_name=self.llm_model_name,
            api_key=self.llm_api_key,
            model_count=self.llm_model_count,
            **self.llm_generate_kwargs
        )
        """初始化实例池：预创建 min_size 个智能体实例（服务启动时调用）"""
        async with self.pool_lock:
            for i in range(self.min_size):
                model, model_lock = await self.llm_pool.acquire_model()
                agent = self._create_agent(model, model_lock)
                self.pool.append(agent)
                # 初始化元数据
                self.agent_meta[agent] = {
                    "session_id": None,
                    "lock": asyncio.Lock(),
                    "last_release_time": datetime.now(),
                    "model_lock": model_lock
                }
            logger.info(
                f"【{self.agent_type_name}实例池】初始化完成，预创建 {self.min_size} 个实例（绑定专属模型池[{self.llm_model_name}]），当前容量：{len(self.pool)}"
            )

    async def _create_agent(self, model: DashScopeChatModel, model_lock: asyncio.Lock) -> AgentType:
        """创建单个智能体实例（内部调用，封装创建逻辑）"""
        try:
            agent = self.agent_creator(model, model_lock)
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

            """
            高并发下，扩容创建的新实例可能被其他会话 “截胡”，导致当前会话陷入 “扩容→被截胡→再扩容→再被截胡” 的循环，极端情况下甚至会触发实例池最大容量限制，最终抛出超时错误。
            """
            # 第二步：无空闲实例，尝试扩容【旧代码】
            # if await self._try_expand_pool():
            #     # 扩容后递归获取（新实例已绑定会话）
            #     return await self.get_agent(session_id)
            # 第二步：无空闲实例，尝试扩容（原子化创建+绑定）【新代码】注意需要将扩容代码以及任何改变实例池中实例数量的操作放置到获取self.pool_lock之下，否则容易出现并发安全问题
            new_agent = await self._try_expand_and_bind(session_id)
            if new_agent:
                logger.info(
                    f"【{self.agent_type_name}实例池】会话 {session_id} 扩容获取新实例 {new_agent.instance_id}"
                )
                return new_agent, new_agent.instance_id

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
            if meta.get("model_lock"):
                self.llm_pool.release_model(meta["model_lock"])  # 释放模型实例锁
                meta["model_lock"] = None
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

    # async def _try_expand_pool(self) -> bool:
    #     """尝试扩容实例池（内部调用）"""
    #     async with self.pool_lock:
    #         current_size = len(self.pool)
    #         if current_size >= self.max_size:
    #             return False  # 已达最大容量，无法扩容
    #
    #         # 创建新实例并绑定会话
    #         new_agent = self._create_agent()
    #         self.pool.append(new_agent)
    #         # 初始化元数据并直接绑定会话
    #         self.agent_meta[new_agent] = {
    #             "session_id": None,  # 后续由 get_agent 绑定
    #             "lock": asyncio.Lock(),
    #             "last_release_time": datetime.now()
    #         }
    #         logger.info(
    #             f"【{self.agent_type_name}实例池】扩容成功，当前容量：{len(self.pool)}/{self.max_size}"
    #         )
    #         return True
    async def _try_expand_and_bind(self, session_id: str) -> Optional[AgentType]:
        """
        当前操作必须放置在获取实例池级锁之下进行，否则容易出现并发安全
            协程 A 判断 len(pool)=19 < 20 → 开始创建；
            协程 B 同时判断 len(pool)=19 < 20 → 也开始创建；
            最终 pool 数量变为 21，突破 max_size 限制；
        原子化操作：创建新实例 + 直接绑定当前会话（避免被截胡）
        仅在池级锁保护下调用
        """
        current_size = len(self.pool)
        if current_size >= self.max_size:
            return None  # 已达最大容量，不扩容

        # 从专属模型池获取模型
        retry_count = 0
        new_agent = None
        model, model_lock = await self.llm_pool.acquire_model()
        while retry_count < 2:
            try:
                new_agent = self._create_agent(model, model_lock)
                break
            except Exception as e:
                retry_count += 1
                logger.error(f"【{self.agent_type_name}】扩容创建实例失败（重试{retry_count}/2）：{str(e)}")
                await asyncio.sleep(0.5)

        if not new_agent:
            return None

        # 绑定会话并加入池
        self.agent_meta[new_agent] = {
            "session_id": session_id,
            "lock": asyncio.Lock(),
            "last_release_time": datetime.now(),
            "model_lock": model_lock
        }
        self.pool.append(new_agent)
        logger.info(
            f"【{self.agent_type_name}】扩容成功，当前容量：{len(self.pool)}/{self.max_size}"
        )
        return new_agent

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
                """
                已经采用了弱引用字典，为何还要显式删除该字典中的智能体实例缓存数据？
                    主要是考虑该智能体实例存在未知的强引用，gc不会回收该智能体实例，也就不会触发弱引用字典对应数据的清理，导致智能体实例的元数据依然存在与agent_meta中，造成实例池管理混乱。
                    但是如果采用普通字段存储智能体实例元数据，会导致删除智能体实例缓存数据的操作完全依赖显式删除，没有任何其他途径。相较于弱引用字典，相当于缺失了一种兜底清理缓存机制。
                我的疑问与推测：
                    疑问：agent空闲时，仅仅是将当前实例从实例池中删除，而该agent实例依然存在，并没有显式的删除该agent实例。
                    推测：是否由于所有的实例获取都是基于实例池进行的，因此对于那些不在实例池中的agent实例【在没有其他强引用时】会被垃圾回收机制回收，
                    【回收时会触发弱引用字典同步删除相应的实例缓存数据，这也是采用弱引用字段的双重保障作用所在】。
                Python 中无法 “显式删除对象实例”（比如没有 delete agent 语法），对象的销毁完全由 “引用计数 + 垃圾回收” 决定：
                    我们能做的只是切断强引用（pool.remove(agent)），让实例的引用计数归 0；
                    当引用计数归 0 时，GC 会自动销毁实例，释放其占用的内存（包括 agent 内部的 model、memory 等资源）；
                    实例池是 agent 实例的主要强引用来源（甚至是唯一来源），只要从 pool 中移除，实例就失去了核心强引用，最终会被 GC 回收。
                为什么不手动 “强制销毁” agent 实例？ 
                    Python 的 GC 是 “自动且高效” 的，手动触发会打断正常的内存管理节奏，增加性能开销；
                    强制 GC 无法解决 “有隐性强引用的实例” 问题（有强引用的实例即使 GC 也不会被回收），反而会掩盖代码中的引用泄漏问题。
                    
                保留弱引用字典的关键原因：
                    即使我们漏做了 del agent_meta[agent]（比如代码异常、逻辑疏漏），只要 agent 实例被 GC 回收，WeakKeyDictionary 会自动、被动地删除对应的元数据；
                    而如果用普通字典，哪怕 agent 实例被 GC 回收，元数据仍会永久留在字典中（因为普通字典的键是强引用，实例销毁不影响字典），直到手动删除或服务重启。
                结论：
                    ✅ 实例池是 agent 实例的核心强引用来源，从 pool 中移除后，无其他强引用的 agent 会被 GC 自动回收；
                    ✅ 弱引用字典的 “双重保障” 体现在：显式 del 保证即时清理，GC 触发时自动兜底清理漏删的元数据；
                    ✅ 不需要（也无法）显式删除 agent 实例，只需切断强引用，GC 会自动完成销毁。
                """
                del self.agent_meta[agent]  # 移除元数据
                # 可选：销毁模型连接（如果智能体持有独立模型）
                agent.model = None
                logger.info(
                    f"【{self.agent_type_name}实例池】缩容销毁实例 {agent.instance_id}，当前容量：{len(self.pool)}"
                )


# -------------------------- 实例池初始化（按智能体类型独立创建） --------------------------
class LLMModelPool:
    """
    改造为非全局单例：每个智能体池可创建专属实例
    保留核心特性：单实例内多模型、LRU负载均衡、并发安全
    """
    def __init__(self):
        """
        模型实例池的核心设计思路：
            模型锁是模型实例使用权的 “唯一凭证”；
            获取模型实例 = 锁定对应的模型锁；
            释放模型实例 = 解锁对应的模型锁；
            智能体绑定锁 = 保存这个 “凭证”，确保释放时能精准归还。
        获取模型实例的同时也获取对应的模型锁，后续之所以要将模型锁与智能体实例绑定并存储，是为了在释放智能体实例时，
        同时释放模型实例【实际是释放了模型锁，使得该锁可以再次被其他协程获取到，也即是创建其他智能体实例可以获取到该锁】。
        总之，模型实例的获取与释放，在模型实例池内部其实是对应模型锁的获取与释放。
        """
        self.models: Optional[List[DashScopeChatModel]] = None
        self.model_locks: Optional[List[asyncio.Lock]] = None
        self.last_used_time: Optional[List[datetime]] = None
        self._init_flag = False
        self._init_lock = asyncio.Lock()

    async def init(
        self,
        model_name: str = "deepseek-v3",
        api_key: str = "sk-6b8afa231399490bb7a56c025a3bc633",
        model_count: int = 3,
        **generate_kwargs
    ):
        """初始化专属模型池（每个智能体池调用一次）"""
        async with self._init_lock:
            if self._init_flag:
                logger.info(f"模型池[{model_name}]已初始化，跳过重复操作")
                return

            default_kwargs = {
                "temperature": 0.1,
                "top_p": 0.8,
                "max_tokens": 300,
                "repetition_penalty": 1.1
            }
            default_kwargs.update(generate_kwargs)

            # 创建专属模型实例列表
            self.models = []
            for i in range(model_count):
                model = DashScopeChatModel(
                    model_name=model_name,
                    api_key=api_key,
                    generate_kwargs=default_kwargs
                )
                model.instance_id = f"{model_name}_model_{i+1}"
                self.models.append(model)

            self.model_locks = [asyncio.Lock() for _ in range(model_count)]
            self.last_used_time = [datetime.now() for _ in range(model_count)]
            self._init_flag = True

            logger.info(f"专属模型池[{model_name}]初始化完成，创建 {model_count} 个实例")

    async def acquire_model(self) -> Tuple[DashScopeChatModel, asyncio.Lock]:
        """获取当前模型池的空闲实例"""
        if not self._init_flag:
            raise RuntimeError("模型池未初始化，请先调用init()")

        while True:
            # LRU负载均衡：优先分配最久未使用的模型
            min_time_idx = self.last_used_time.index(min(self.last_used_time))
            target_lock = self.model_locks[min_time_idx]

            # 尝试获取锁（超时1秒，避免死等）
            try:
                if await asyncio.wait_for(target_lock.acquire(), timeout=1):  # 在获取模型实例时，同步已经获取了模型锁。
                    self.last_used_time[min_time_idx] = datetime.now()
                    return self.models[min_time_idx], target_lock
            except asyncio.TimeoutError:
                pass

            # 所有模型忙，短暂等待后重试
            await asyncio.sleep(0.1)

    def release_model(self, lock: asyncio.Lock):
        """释放模型锁"""
        if lock and lock.locked():
            lock.release()
            logger.debug("模型锁已释放")

    @property
    def is_initialized(self) -> bool:
        """检查模型池是否已初始化"""
        return self._init_flag


# ===================== 全局初始化（多智能体池+专属模型池） =====================
async def init_all_agent_pools() -> Tuple[GenericAgentPool, GenericAgentPool, GenericAgentPool]:
    """
    初始化所有智能体池，每个池绑定专属模型池：
    1. 重写助手：deepseek-v3，3个模型实例
    2. 检索助手：deepseek-v3，5个模型实例（检索耗时更长，需要更多并发）
    3. 运维专家：gpt-4，2个模型实例（高端模型，控制并发）
    """
    # 1. 重写助手池（绑定专属模型池）
    rewrite_pool = GenericAgentPool(
        agent_creator=create_rewrite_agent,
        agent_type_name="重写助手",
        llm_model_name="deepseek-v3",
        llm_api_key="sk-6b8afa231399490bb7a56c025a3bc633",
        llm_model_count=3,
        llm_generate_kwargs={"temperature": 0.1},
        min_size=5,
        max_size=30
    )

    # 2. 检索助手池（绑定专属模型池，更多模型实例）
    retriever_pool = GenericAgentPool(
        agent_creator=create_retriever_agent,
        agent_type_name="检索助手",
        llm_model_name="deepseek-v3",
        llm_api_key="sk-6b8afa231399490bb7a56c025a3bc633",
        llm_model_count=5,  # 检索耗时更长，增加模型实例数
        llm_generate_kwargs={"temperature": 0.0, "max_tokens": 500},
        min_size=5,
        max_size=30
    )

    # 3. 运维专家池（绑定专属高端模型池）
    expert_pool = GenericAgentPool(
        agent_creator=create_expert_agent,
        agent_type_name="运维专家",
        llm_model_name="deepseek-v3",  # 不同模型
        llm_api_key="sk-6b8afa231399490bb7a56c025a3bc633",  # 不同API Key
        llm_model_count=2,  # 控制高端模型并发数
        llm_generate_kwargs={"temperature": 0.2, "max_tokens": 1000},
        min_size=2,
        max_size=15
    )

    # 初始化所有池
    await rewrite_pool.init_pool()
    await retriever_pool.init_pool()
    await expert_pool.init_pool()

    # 启动后台清理任务
    asyncio.create_task(rewrite_pool.start_cleanup_task())
    asyncio.create_task(retriever_pool.start_cleanup_task())
    asyncio.create_task(expert_pool.start_cleanup_task())

    logger.info("所有智能体池（含专属模型池）初始化完成！")
    return rewrite_pool, retriever_pool, expert_pool
