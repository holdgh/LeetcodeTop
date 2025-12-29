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

from agent_work.agent_scope.agent.rewrite_agent import create_input_text_for_rewrite
from agent_work.agent_scope.agent_pool.agent_pool import AgentPool, SESSION_GOING_CACHE
from agent_work.agent_scope.agent.search_agent import TOOL_CALL_CACHE
from agent_work.database.context_service import get_session_history_context
from agent_work.datatransfer.async_memory_writer import send_message_to_queue_by_async

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# 新增缓存锁
SESSION_CACHE_LOCK = asyncio.Lock()


# 会话管理--会话id生成
class SessionManager:
    @staticmethod
    def create_session() -> str:
        """创建新会话ID"""
        return f"session_{uuid.uuid4().hex[:8]}"


async def user_dialog_for_one_question(session_id: str, agent_pool: AgentPool, question: str, conversation_id: str,
                                       user_id: str = "default_user"):
    """单个用户多轮对话（复用Agent实例）"""
    logger.info(f"\n【会话 {session_id}】用户开始咨询")
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
        user_msg = Msg(name="user", content=create_input_text_for_rewrite(current_question=question, history_context=history_context), role="user", invocation_id=conversation_id)
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
        await agent_pair.unbind_session()
        # 6. 输出结果
        logger.info(f"【会话 {session_id}】收到回复：{expert_reply.content}\n")
        return expert_reply.content
    except Exception as e:
        traceback.print_exc()
        logger.error(f"【会话 {session_id}】处理异常：{str(e)}")


# -------------------------- 5. 服务端 API 与业务逻辑 --------------------------
# 请求/响应模型
class QueryRequest(BaseModel):
    session_id: Optional[str] = None  # 会话ID（首次请求可空，自动创建）
    question: str  # 用户问题


class QueryResponse(BaseModel):
    session_id: str  # 会话ID（用于多轮对话）
    reply: str  # 运维专家回复
    timestamp: datetime  # 响应时间


# 全局实例池（启动时初始化）
agent_pool = AgentPool(min_size=2, max_size=4)


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
    async with SESSION_CACHE_LOCK:
        is_going = SESSION_GOING_CACHE.get(session_id, 0)
        if is_going == 1:
            return QueryResponse(
                session_id=session_id,
                reply="当前对话尚未结束，请待当前对话完成后进行！",
                timestamp=datetime.now()
            )
        SESSION_GOING_CACHE[session_id] = 1  # 标记为进行中
    # 每次对话生成对话id
    conversation_id = f"conversation_{uuid.uuid4().hex[:8]}"
    try:
        # 获取对话结果
        result = await user_dialog_for_one_question(session_id, agent_pool, request.question, conversation_id, user_id)
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
        if session_id in SESSION_GOING_CACHE:
            del SESSION_GOING_CACHE[session_id]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app='api_plus_backup_for_agent_pair:app', port=8090, reload=False,
                workers=1)
