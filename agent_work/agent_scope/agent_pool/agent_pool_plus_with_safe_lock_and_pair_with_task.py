from datetime import datetime, timedelta
from enum import Enum
import asyncio
from typing import List, Optional
from agentscope.agent import ReActAgent
from agentscope.model import DashScopeChatModel
# import logging
from cachetools import TTLCache

from agent_work.agent_scope.agent.expert_agent import create_expert_agent
from agent_work.agent_scope.agent.rewrite_agent import create_rewrite_agent
from agent_work.agent_scope.agent.search_agent import create_retriever_agent

# 日志配置
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# 采用日志工具类--同时输出日志到控制台和文件
from agent_work.util.logger import get_logger

logger = get_logger("agent_work_agent_pool")
# 缓存配置：key=会话id，value=当前会话是否进行中的标识
SESSION_GOING_CACHE = TTLCache(maxsize=1000, ttl=1800)


# -------------------------- Agent实例状态枚举 --------------------------
class AgentInstanceState(Enum):
    IDLE = "空闲"
    BUSY = "忙碌"
    # 核心改造
    PENDING_DELETE = "pending_delete"  # 待删除（清理任务标记，禁止被占用）
    DELETED = "deleted"  # 已删除（最终状态，仅用于日志追溯）


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
        self.pending_delete = False  # 是否待清理
        self.instance_lock = asyncio.Lock()  # 仅在实例进行状态判断、会话绑定时持有该锁
        self.retriever = self._create_retriever_agent()
        self.expert = self._create_expert_agent()
        self.rewriter = self._create_rewrite_agent()
        self._current_task = None  # 私有属性，避免外部随意修改

        # 封装current_task的读写，确保任务结束后自动清空

    @property
    def current_task(self):
        return self._current_task

    @current_task.setter
    def current_task(self, task):
        self._current_task = task
        # 任务结束后自动清空current_task
        if task:
            def task_done_callback(fut):
                self._current_task = None  # 任务完成，自动清空

            task.add_done_callback(task_done_callback)

    def _create_retriever_agent(self) -> ReActAgent:
        return create_retriever_agent(self.model)

    def _create_expert_agent(self) -> ReActAgent:
        return create_expert_agent(self.model)

    def _create_rewrite_agent(self) -> ReActAgent:
        return create_rewrite_agent(self.model)

    def get_state(self) -> AgentInstanceState:
        if self.pending_delete:
            return AgentInstanceState.PENDING_DELETE
        if self.session_id is None:
            return AgentInstanceState.IDLE
        return AgentInstanceState.BUSY

    # async def set_state(self, new_state):
    #     """原子修改状态（必须加锁，且校验流转规则）"""
    #     async with self.instance_lock:
    #         # 校验状态流转合法性（避免非法状态变更）当前状态-->目标状态
    #         valid_transitions = {
    #             AgentInstanceState.IDLE: [AgentInstanceState.BUSY, AgentInstanceState.PENDING_DELETE],  # 新会话占用、空闲实例清理标记
    #             AgentInstanceState.BUSY: [AgentInstanceState.IDLE],  # 会话释放实例
    #             AgentInstanceState.PENDING_DELETE: [AgentInstanceState.IDLE, AgentInstanceState.DELETED],  # 取消删除、实例已删除
    #             AgentInstanceState.DELETED: []
    #         }
    #         if new_state not in valid_transitions[self._state]:
    #             raise ValueError(
    #                 f"实例 {self.pair_id} 状态流转非法：{self._state.value} → {new_state.value}"
    #             )
    #         self._state = new_state
    #         logger.debug(f"实例 {self.pair_id} 状态更新：{self._state.value}")

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
            # 第一步：终止未完成的任务（核心）
            if self.current_task and not self.current_task.done():
                # 协程任务取消机制：task.cancel()是“异步指令”，调用后任务不会立即停止，await task的作用：阻塞到任务真正终止（无论是正常结束还是被取消）
                self.current_task.cancel()
                try:
                    await self.current_task  # 执行这行时，任务会抛出CancelledError，被try捕获
                except asyncio.CancelledError:
                    pass
            # 第二步：清除各智能体实例缓存
            await self.retriever.memory.clear()
            await self.expert.memory.clear()
            await self.rewriter.memory.clear()
            # if self.session_id in SESSION_GOING_CACHE:
            #     del SESSION_GOING_CACHE[self.session_id]
            # 第三步：解除业务会话关联
            self.session_id = None
            logger.info(f"Agent实例对 {self.pair_id} 解除会话绑定，回归空闲")

        # 触发空闲通知（需将agent_pool设为全局/类属性，或通过参数传入）
        global agent_pool
        await agent_pool.notify_idle()


# -------------------------- Agent实例池 --------------------------
async def _clean_agent_pair_resources(agent_pair: AgentPair):
    """清理AgentPair的资源（避免泄漏）"""
    # 主动置空实例的核心属性，断除内部引用链
    agent_pair.retriever = None
    agent_pair.expert = None
    agent_pair.rewriter = None
    agent_pair.model = None
    agent_pair.current_task(None)
    agent_pair.session_id = None
    agent_pair.pair_id = None
    agent_pair.pending_delete = None


class AgentPool:
    def __init__(self, min_size: int = 5, max_size: int = 20):
        self.min_size = min_size
        self.max_size = max_size
        self.pool: List[AgentPair] = []
        self.lock = asyncio.Lock()  # 仅在改变实例池时持有该锁
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

    # async def _wait_for_idle_pair(self, session_id: str, timeout: int = 60) -> AgentPair:
    #     """等待空闲实例（条件变量替代轮询）"""
    #     start_time = datetime.now()
    #     async with self.idle_condition:
    #         while True:
    #             if datetime.now() - start_time > timedelta(seconds=timeout):
    #                 raise TimeoutError("当前咨询用户过多，请稍后重试")
    #
    #             # 尝试绑定空闲实例
    #             idle_pairs = [p for p in self.pool if p.get_state() == AgentInstanceState.IDLE]
    #             if idle_pairs:
    #                 for agent_pair in idle_pairs:
    #                     try:
    #                         # 步骤1：快照校验：确认实例当前是否仍为空闲（避免脏读）
    #                         async with agent_pair.instance_lock:
    #                             if agent_pair.get_state() == AgentInstanceState.IDLE:
    #                                 # 步骤2：尝试绑定会话（可能触发“已绑定”异常）
    #                                 # await agent_pair.bind_session(session_id)
    #                                 agent_pair.session_id = session_id
    #                                 # 绑定成功 → 返回实例
    #                                 return agent_pair
    #                     except ValueError as e:
    #                         # 捕获“实例已被绑定”的异常（竞争失败），继续尝试下一个空闲实例
    #                         logger.warning(f"实例 {agent_pair.pair_id} 竞争失败（{str(e)}），尝试下一个实例")
    #                         continue  # 跳过当前实例，遍历下一个
    #
    #             # 无空闲，挂起协程等待通知（最多等5秒，避免永久阻塞）
    #             try:
    #                 async with self.idle_condition:
    #                     await asyncio.wait_for(self.idle_condition.wait(), timeout=5)
    #             except asyncio.TimeoutError:
    #                 # 5秒无通知，自动唤醒重新检查（兜底）
    #                 continue
    async def _wait_for_idle_pair(self, session_id: str, timeout: int = 60) -> AgentPair:
        """等待空闲实例（修复：移除嵌套condition，加超时释放锁）"""
        start_time = datetime.now()
        # 1. 获取self.lock（锁状态：locked）
        async with self.idle_condition:
            while True:
                # 超时判断（优先释放锁）
                if datetime.now() - start_time > timedelta(seconds=timeout):
                    # 2. 抛出异常 → 跳出async with → 释放self.lock（锁状态：unlocked）
                    raise TimeoutError("当前咨询用户过多，请稍后重试")

                # 锁内筛选空闲实例（避免脏读，锁状态：locked）
                idle_pairs = [p for p in self.pool if p.get_state() == AgentInstanceState.IDLE]
                if idle_pairs:
                    for agent_pair in idle_pairs:
                        try:
                            async with agent_pair.instance_lock:
                                if agent_pair.get_state() == AgentInstanceState.IDLE:
                                    agent_pair.session_id = session_id
                                    agent_pair.current_task = asyncio.current_task()
                                    # 3. 返回实例 → 跳出async with → 释放self.lock（锁状态：unlocked）
                                    return agent_pair
                        except Exception as e:
                            logger.warning(f"实例 {agent_pair.pair_id} 竞争失败：{str(e)}")
                            continue

                # 4. 无空闲实例，执行wait() → 自动释放self.lock（锁状态：unlocked），挂起协程
                try:
                    # wait 5秒后自动唤醒，避免永久阻塞
                    await asyncio.wait_for(self.idle_condition.wait(), timeout=5)
                    # 5. 唤醒后 → 自动重新获取self.lock（锁状态：locked），继续循环
                except asyncio.TimeoutError:
                    # 超时唤醒 → 重新获取self.lock（锁状态：locked），继续循环
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

    # async def clean_expired_pairs(self):
    #     """定期清理过期实例（优化：锁粒度最小化，避免阻塞整个实例池）"""
    #     try:
    #         while True:
    #             await asyncio.sleep(30)
    #             logger.info(f"开始清理过期实例，当前池容量：{len(self.pool)}")
    #             # 核心优化：用asyncio.wait_for确保sleep不被阻塞（超时时间略大于30秒）
    #             # try:
    #             #     await asyncio.wait_for(
    #             #         asyncio.sleep(30),
    #             #         timeout=35  # 30秒sleep最多等35秒，超时强制唤醒
    #             #     )
    #             # except asyncio.TimeoutError:
    #             #     logger.warning("清理任务sleep超时，强制唤醒执行")
    #
    #             # ========== 步骤1：无锁快照筛选疑似空闲实例（无池级锁，不阻塞任何操作） ==========
    #             idle_pairs = [p for p in self.pool if p.get_state() == AgentInstanceState.IDLE]
    #
    #             if not idle_pairs:
    #                 logger.info("无疑似空闲实例，跳过清理")
    #                 continue
    #
    #             # ========== 步骤2：加实例锁确认真·空闲实例（无池级锁） ==========
    #             real_idle_pairs = []
    #             for p in idle_pairs:
    #                 try:
    #                     async with p.instance_lock:
    #                         # 锁内最终确认是否空闲且待清理
    #                         if p.get_state() == AgentInstanceState.IDLE:
    #                             p.pending_delete = True  # 标记待清理，以避免新会话占用
    #                             real_idle_pairs.append(p)
    #                 except Exception as e:
    #                     logger.warning(f"校验实例 {p.pair_id} 状态失败：{str(e)}")
    #                     continue
    #
    #             # ========== 步骤3：仅在删除实例时加池级锁（最小化持有时间） ==========
    #             idle_count = len(real_idle_pairs)
    #             # 缩容阈值：仅当空闲实例数 > min_size 且 池总容量 > min_size 时缩容
    #             if idle_count > self.min_size and len(self.pool) > self.min_size:
    #                 # 计算需要删除的实例数：保留min_size个空闲实例
    #                 need_delete_count = idle_count - self.min_size
    #                 # 取前need_delete_count个空闲实例（避免删除所有）
    #                 redundant_pairs = real_idle_pairs[:need_delete_count]
    #                 # 保留的空闲实例（需重置标记）
    #                 reserve_pairs = real_idle_pairs[need_delete_count:]
    #                 # 加池级锁删除实例（仅这一步持有池级锁，耗时极短）
    #                 async with self.lock:
    #                     deleted_count = 0
    #                     for p in redundant_pairs:
    #                         if p in self.pool:  # 二次校验，避免重复删除
    #                             # 清理实例资源（核心：避免内存/连接泄漏）
    #                             await _clean_agent_pair_resources(p)
    #                             self.pool.remove(p)
    #                             deleted_count += 1
    #
    #                 logger.info(
    #                     f"实例池缩容完成，删除 {deleted_count} 个空闲实例，当前容量：{len(self.pool)}，剩余空闲数：{idle_count - deleted_count}")
    #                 # 重置保留实例的待清理标记（关键：恢复可用）【必须放置在删除实例之后，以保留实例池清理逻辑的原子性【清理多余的实例，而不是清理实例。需要保持一定数量的空闲实例。】】
    #                 for p in reserve_pairs:
    #                     try:
    #                         async with p.instance_lock:
    #                             p.pending_delete = False
    #                     except Exception as e:
    #                         logger.warning(f"重置实例 {p.pair_id} 待清理标记失败：{str(e)}")
    #             else:
    #                 logger.info(f"无需缩容，当前空闲实例数：{idle_count}，最小空闲数：{self.min_size}")
    #                 # 当空闲实例不足以达到清理条件时，将当前所有待清理实例恢复可用状态
    #                 for p in real_idle_pairs:
    #                     try:
    #                         async with p.instance_lock:
    #                             p.pending_delete = False
    #                     except Exception as e:
    #                         logger.warning(f"重置实例 {p.pair_id} 待清理标记失败：{str(e)}")
    #
    #     except asyncio.CancelledError:
    #         logger.info("清理过期实例的后台任务已被取消")
    #         return
    #     except Exception as e:
    #         logger.error(f"清理过期实例时发生异常：{str(e)}", exc_info=True)

    async def clean_expired_pairs(self):
        """定期清理过期实例（优化：锁粒度最小化+异常释放锁）"""
        try:
            while True:
                await asyncio.sleep(30)
                logger.info(f"开始清理过期实例，当前池容量：{len(self.pool)}")

                # 步骤1：无锁快照筛选疑似空闲实例
                idle_pairs = [p for p in self.pool if p.get_state() == AgentInstanceState.IDLE]
                if not idle_pairs:
                    logger.info("无疑似空闲实例，跳过清理")
                    continue

                # 步骤2：加实例锁确认真·空闲实例（带finally释放）
                real_idle_pairs = []
                for p in idle_pairs:
                    instance_lock_acquired = False
                    try:
                        # 实例锁加超时，避免卡死
                        await asyncio.wait_for(p.instance_lock.acquire(), timeout=2)
                        instance_lock_acquired = True
                        if p.get_state() == AgentInstanceState.IDLE:
                            p.pending_delete = True
                            real_idle_pairs.append(p)
                    except asyncio.TimeoutError:
                        logger.warning(f"获取实例 {p.pair_id} 锁超时，跳过")
                        continue
                    except Exception as e:
                        logger.warning(f"校验实例 {p.pair_id} 状态失败：{str(e)}")
                        continue
                    finally:
                        # 确保实例锁释放
                        if instance_lock_acquired:
                            p.instance_lock.release()

                # 步骤3：仅删除时加池锁（最小化持有时间）
                idle_count = len(real_idle_pairs)
                if idle_count > self.min_size and len(self.pool) > self.min_size:
                    need_delete_count = idle_count - self.min_size
                    redundant_pairs = real_idle_pairs[:need_delete_count]
                    reserve_pairs = real_idle_pairs[need_delete_count:]

                    # 加池锁删除实例（带超时）
                    pool_lock_acquired = False
                    try:
                        await asyncio.wait_for(self.lock.acquire(), timeout=5)
                        pool_lock_acquired = True
                        deleted_count = 0
                        for p in redundant_pairs:
                            if p in self.pool:
                                await _clean_agent_pair_resources(p)
                                self.pool.remove(p)
                                deleted_count += 1
                        logger.info(
                            f"实例池缩容完成，删除 {deleted_count} 个空闲实例，当前容量：{len(self.pool)}")
                    except asyncio.TimeoutError:
                        logger.error("获取池锁超时，缩容失败")
                    finally:
                        if pool_lock_acquired:
                            self.lock.release()

                    # 重置保留实例标记（带finally释放）
                    for p in reserve_pairs:
                        instance_lock_acquired = False
                        try:
                            await asyncio.wait_for(p.instance_lock.acquire(), timeout=2)
                            instance_lock_acquired = True
                            p.pending_delete = False
                        except Exception as e:
                            logger.warning(f"重置实例 {p.pair_id} 标记失败：{str(e)}")
                        finally:
                            if instance_lock_acquired:
                                p.instance_lock.release()
                else:
                    logger.info(f"无需缩容，当前空闲实例数：{idle_count}")
                    # 重置待清理标记（带finally释放）
                    for p in real_idle_pairs:
                        instance_lock_acquired = False
                        try:
                            await asyncio.wait_for(p.instance_lock.acquire(), timeout=2)
                            instance_lock_acquired = True
                            p.pending_delete = False
                        except Exception as e:
                            logger.warning(f"重置实例 {p.pair_id} 标记失败：{str(e)}")
                        finally:
                            if instance_lock_acquired:
                                p.instance_lock.release()

        except asyncio.CancelledError:
            logger.info("清理过期实例的后台任务已被取消")
            return
        except Exception as e:
            logger.error(f"清理过期实例异常：{str(e)}", exc_info=True)

    # async def notify_idle(self):
    #     """触发空闲实例通知"""
    #     async with self.idle_condition:
    #         self.idle_condition.notify_all()
    async def notify_idle(self):
        """触发空闲实例通知（修复：加超时+finally释放锁）"""
        lock_acquired = False
        try:
            # 加超时，避免卡死在获取锁
            await asyncio.wait_for(self.idle_condition.acquire(), timeout=3)
            lock_acquired = True
            self.idle_condition.notify_all()
            logger.debug("触发空闲实例通知，唤醒所有等待协程")
        except asyncio.TimeoutError:
            logger.warning("获取idle_condition锁超时，无法触发通知")
        finally:
            if lock_acquired:
                self.idle_condition.release()


# 全局实例池（供unbind_session调用）
agent_pool = AgentPool(min_size=5, max_size=20)
