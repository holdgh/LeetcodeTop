import threading
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
import asyncio
from typing import Optional
from agentscope.message import Msg
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from cachetools import TTLCache

from agent_work.agent_scope.agent.rewrite_agent import create_input_text_for_rewrite
from agent_work.agent_scope.agent_pool.agent_pool_plus import SESSION_GOING_CACHE, agent_pool
from agent_work.agent_scope.agent.search_agent import TOOL_CALL_CACHE
from agent_work.database.context_service import get_session_history_context
from agent_work.datatransfer.async_memory_writer import send_message_to_queue_by_async

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# 新增缓存锁
SESSION_CACHE_LOCK = asyncio.Lock()


# ===================== 新增：全局会话控制器（单例） =====================


class GlobalSessionController:
    """
    全局会话控制器：基于TTLCache实现（极简+稳定）
    核心特性：
    1. 单例模式
    2. 原子化锁定/解锁（基于TTLCache的线程/协程安全操作）
    3. 内置自动过期+最大容量，无需手动清理
    4. 无异步任务/循环，彻底避免卡住问题
    """
    _instance: Optional["GlobalSessionController"] = None
    _instance_lock = threading.Lock()  # 保护单例创建（线程安全）
    # 核心：用TTLCache存储会话锁定状态
    # key=session_id，value=True（仅标记“已锁定”，无需存储其他状态）
    # maxsize：最大并发会话数；ttl：会话锁定自动过期时间（5分钟）
    _session_cache: Optional[TTLCache] = None

    def __new__(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    # 初始化TTLCache（核心配置）
                    cls._instance._session_cache = TTLCache(
                        maxsize=1000,  # 最大并发会话数限制
                        ttl=300,  # 会话锁定5分钟自动过期（兜底）
                    )
        return cls._instance

    async def lock_session(self, session_id: str) -> bool:
        """
        锁定会话（原子操作）
        :return: True=锁定成功，False=会话已锁定/超过最大容量
        """
        # TTLCache的操作是线程/协程安全的，无需额外加锁
        if session_id in self._session_cache:
            logger.warning(f"【会话控制】会话[{session_id}]已锁定，锁定失败")
            return False

        # 检查是否超过最大容量（可选：TTLCache会自动淘汰，但提前判断更友好）
        if len(self._session_cache) >= self._session_cache.maxsize:
            logger.warning(f"【会话控制】会话[{session_id}]锁定失败：超过最大并发数{self._session_cache.maxsize}")
            return False

        # 锁定会话（存储True，标记已锁定）
        self._session_cache[session_id] = True
        logger.info(f"【会话控制】会话[{session_id}]锁定成功，当前会话数：{len(self._session_cache)}")
        return True

    async def unlock_session(self, session_id: str) -> None:
        """解锁会话（原子操作）"""
        if session_id in self._session_cache:
            del self._session_cache[session_id]
            logger.info(f"【会话控制】会话[{session_id}]解锁成功，当前会话数：{len(self._session_cache)}")
        else:
            logger.warning(f"【会话控制】会话[{session_id}]未锁定，无需解锁")

    async def force_unlock(self, session_id: str) -> None:
        """强制解锁（异常兜底）"""
        if session_id in self._session_cache:
            del self._session_cache[session_id]
            logger.warning(f"【会话控制】会话[{session_id}]已强制解锁")


# 会话管理--会话id生成
class SessionManager:
    @staticmethod
    def create_session() -> str:
        """创建新会话ID"""
        return f"session_{uuid.uuid4().hex[:8]}"


async def user_dialog_for_one_question(session_id: str, question: str, conversation_id: str,
                                       user_id: str = "default_user"):
    """单个用户多轮对话（复用Agent实例）"""
    logger.info(f"\n【会话 {session_id}】用户开始咨询")
    agent_pair = None
    try:
        # 1. 获取绑定该会话的Agent实例对
        agent_pair = await agent_pool.get_agent_pair(session_id)
        rewriter = agent_pair.rewriter
        retriever = agent_pair.retriever
        expert = agent_pair.expert
        # 发送用户消息
        await send_message_to_queue_by_async(
            user_id=user_id,
            session_id=session_id,
            conversation_id=conversation_id,
            role="user",
            content=question
        )
        logger.info(f"【会话 {session_id}】发送：{question}")
        # 2. 获取会话历史上下文
        # 获取相应会话历史上下文
        history_context, summary = await get_session_history_context(session_id)
        # 整合用户消息与历史上下文
        user_msg = Msg(name="user", content=create_input_text_for_rewrite(current_question=question,
                                                                          history_context=history_context), role="user",
                       invocation_id=conversation_id)
        # 3. 重写问题
        # 将重写信息给到重写助手
        await rewriter.memory.add(user_msg)
        # 获取重写助手结果
        rewriter_reply = await rewriter.reply()
        # 发送重写助手消息
        await send_message_to_queue_by_async(
            user_id=user_id,
            session_id=session_id,
            conversation_id=conversation_id,
            role="rewriter",
            content=rewriter_reply.content
        )
        logger.info(f"【会话 {session_id}】的重写助手返回数据：{rewriter_reply.content}")
        # 依据重写问题构造新的用户信息，用以给到检索助手智能体处理
        user_rewrite_msg = Msg(name="user", content=rewriter_reply.content, role="user", invocation_id=conversation_id)
        # 4. 将重写后的问题存入实例专属内存
        await retriever.memory.add(user_rewrite_msg)
        # 直接将重写助手回复给到运维专家，用以告知运维专家当前消息是重写后的问题
        await expert.memory.add(rewriter_reply)
        # 构造历史对话摘要信息，帮助运维专家更好地回答问题
        await expert.memory.add(Msg(name="历史对话摘要", content=summary, role="system", invocation_id=conversation_id))
        # 5. 智能体协作处理
        retriever_reply = await retriever.reply()
        # 发送检索助手消息
        await send_message_to_queue_by_async(
            user_id=user_id,
            session_id=session_id,
            conversation_id=conversation_id,
            role="retriever",
            content=str(retriever_reply.content)  # 转换为字符串存储
        )
        logger.info(f"【会话 {session_id}】的检索助手返回数据：{retriever_reply.content}")
        await expert.memory.add(retriever_reply)
        expert_reply = await expert.reply()
        # 发送运维专家消息
        await send_message_to_queue_by_async(
            user_id=user_id,
            session_id=session_id,
            conversation_id=conversation_id,
            role="expert",
            content=expert_reply.content
        )
        # -------------------------------------------------------------------------------------
        # TODO 在得到运维专家回复后，清除智能体实例中的缓存数据【历史对话及工作调用数据】，并将当前会话与当前智能体实例接触绑定，后续新对话进来重新从资源池获取新的智能体实例
        # await agent_pair.unbind_session()  # 将其放置在finally中，更稳定
        # 6. 输出结果
        logger.info(f"【会话 {session_id}】收到回复：{expert_reply.content}\n")
        return expert_reply.content
    except Exception as e:
        traceback.print_exc()
        logger.error(f"【会话 {session_id}】处理异常：{str(e)}")
    finally:
        # ========== 核心补偿机制：无论是否异常，都释放实例 ==========
        if agent_pair:
            try:
                # 第一步：调用正常解绑逻辑
                await agent_pair.unbind_session()
                logger.info(f"会话 {session_id} 正常释放实例 {agent_pair.pair_id}")
            except Exception as e:
                # 第二步：解绑失败，强制兜底清空session_id（关键容错）
                logger.warning(f"会话 {session_id} 正常解绑失败，执行强制释放：{str(e)}")
                async with agent_pair.instance_lock:
                    await agent_pair.retriever.memory.clear()
                    await agent_pair.expert.memory.clear()
                    await agent_pair.rewriter.memory.clear()
                    agent_pair.session_id = None

                await agent_pool.notify_idle()
                logger.info(f"会话 {session_id} 强制释放实例 {agent_pair.pair_id} 成功")
        # 解锁当前会话
        await session_controller.unlock_session(session_id)


# -------------------------- 5. 服务端 API 与业务逻辑 --------------------------
# 请求/响应模型
class QueryRequest(BaseModel):
    session_id: Optional[str] = None  # 会话ID（首次请求可空，自动创建）
    question: str  # 用户问题


class QueryResponse(BaseModel):
    session_id: str  # 会话ID（用于多轮对话）
    reply: str  # 运维专家回复
    timestamp: datetime  # 响应时间


@asynccontextmanager
async def init_setting(app: FastAPI):
    """
    生命周期事件lifespan：
        - 您可以定义应用程序启动之前应执行的逻辑（代码）。这意味着在应用程序开始接收请求之前，此代码将执行一次。

        - 同样，您可以定义应用程序关闭时应执行的逻辑（代码）。在这种情况下，在处理了可能的许多请求后，此代码将执行一次。

    使用注意事项：
        - 必须包含 yield：@asynccontextmanager 装饰的函数需要是异步生成器，通过 yield 语句将函数分为两部分：
            -- yield 之前：服务启动时执行（替代原来的 startup 事件）；
            -- yield 之后：服务关闭时执行（替代原来的 shutdown 事件，可选）。
        - 后台任务的优雅关闭：保存 asyncio.create_task 返回的任务对象，在 yield 之后通过 cancel() 取消任务并等待结束，避免服务退出时残留未完成的任务。
    """
    """服务启动时初始化实例池和后台任务"""
    await agent_pool.init_pool()
    # 启动后台任务（用变量保存任务，方便后续关闭）
    clean_task = asyncio.create_task(agent_pool.clean_expired_pairs())
    logger.info("服务启动完成，等待请求...")

    yield  # 关键：分割启动和关闭逻辑，程序会在此时开始处理请求

    # 关闭逻辑（服务退出时执行，可选）
    clean_task.cancel()  # 取消后台任务
    await clean_task  # 等待任务结束
    logger.info("服务已关闭，资源已释放")


# -------------------------- 全局配置 --------------------------
# 初始化全局会话控制器
session_controller = GlobalSessionController()
app = FastAPI(title="工业设备运维Agent服务端", version="1.0", lifespan=init_setting)

# CORS配置（允许前端跨域）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """处理用户查询：获取Agent实例→生成回复→返回结果"""
    # 生成会话ID（首次请求）
    session_id = request.session_id or SessionManager.create_session()
    # 此处user_id为默认值，若有用户登录系统，可从Token中解析真实user_id
    user_id = "default_user"  # 替换为实际用户ID（如从请求头Token提取
    # TODO 风险点2
    """
    SESSION_GOING_CACHE的操作非原子性，存在并发安全问题
        handle_query中判断会话是否进行中的逻辑：
            is_going = SESSION_GOING_CACHE.get(session_id, 0)
            if is_going == 1:
                return  # 提示“对话未结束”
            SESSION_GOING_CACHE.setdefault(session_id, 1)  # 标记为进行中
        get和setdefault是两个独立操作，无锁保护 —— 若两个协程同时查询同一未标记的session_id，均会判断is_going=0，进而同时标记为1，导致同一会话的并发请求绕过拦截，引发后续AgentPair处理冲突。
    """
    # 校验当前会话是否正在进行中
    # is_going = SESSION_GOING_CACHE.get(session_id, 0)
    # if is_going == 1:
    #     return QueryResponse(
    #         session_id=session_id,
    #         reply="当前对话尚未结束，请待当前对话完成后进行！",
    #         timestamp=datetime.now()
    #     )
    # 新代码如下:
    # 原子性判断并标记会话状态
    # async with SESSION_CACHE_LOCK:
    #     is_going = SESSION_GOING_CACHE.get(session_id, 0)
    #     if is_going == 1:
    #         return QueryResponse(
    #             session_id=session_id,
    #             reply="当前对话尚未结束，请待当前对话完成后进行！",
    #             timestamp=datetime.now()
    #         )
    #     SESSION_GOING_CACHE[session_id] = 1  # 标记为进行中
    # 采用全局会话缓存控制，易维护
    not_lock = await session_controller.lock_session(session_id)
    if not not_lock:  # 锁定失败，说明会话处于进行中状态
        return QueryResponse(
            session_id=session_id,
            reply="当前对话尚未结束或系统会话容量已达上限，请稍后重试！",
            timestamp=datetime.now()
        )
    # 每次对话生成对话id
    conversation_id = f"conversation_{uuid.uuid4().hex[:8]}"
    try:
        # 获取对话结果
        result = await user_dialog_for_one_question(session_id, request.question, conversation_id, user_id)
        # 返回结果
        return QueryResponse(
            session_id=session_id,
            reply=result,
            timestamp=datetime.now()
        )

    except TimeoutError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"处理请求异常：{str(e)}")
        raise HTTPException(status_code=500, detail="服务器内部错误")
    finally:
        # 对话结束，清理缓存（无论成功/失败都执行）
        if conversation_id in TOOL_CALL_CACHE:
            del TOOL_CALL_CACHE[conversation_id]
        # 对话结束，清理缓存，相当于将当前会话置为空闲状态
        # if session_id in SESSION_GOING_CACHE:
        #     del SESSION_GOING_CACHE[session_id]
        # 采用全局会话缓存控制，易维护
        await session_controller.force_unlock(session_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app='api_plus_for_agent_scope_for_true_scene_multi_person_and_talk_demo:app', port=8090, reload=False,
                workers=1)
