import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import os
from typing import List, Dict, Optional
from agentscope.agent import ReActAgent
from agentscope.message import Msg, TextBlock, ToolUseBlock, ToolResultBlock
from agentscope.model import DashScopeChatModel
from agentscope.formatter import DashScopeMultiAgentFormatter
from sentence_transformers import SentenceTransformer
from agentscope.tool import Toolkit, ToolResponse
import faiss
import pickle
import logging
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from cachetools import TTLCache  # 带过期时间的缓存，避免内存泄漏

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# 缓存配置：key=dialog_msg_id（对话消息ID），value=工具调用次数，过期时间=30分钟（覆盖单次对话最大耗时）
TOOL_CALL_CACHE = TTLCache(maxsize=1000, ttl=1800)  # 最多缓存1000个对话，30分钟无操作自动过期
MAX_TOOL_CALLS_PER_DIALOG = 3
# 新增缓存锁
SESSION_CACHE_LOCK = asyncio.Lock()
# 缓存配置：key=会话id，value=当前会话是否进行中的标识，0标识空闲，1表示进行中，30分钟误操作自动过期
SESSION_GOING_CACHE = TTLCache(maxsize=1000, ttl=1800)  # 最多缓存1000个会话，30分钟无操作自动过期
from functools import wraps
from typing import Dict, Any, Callable, Optional
import inspect


def validate_params(validators: Dict[str, Callable[[Any], bool]]) -> Any:
    """
    一个通用的参数校验装饰器。

    Args:
        validators (Dict[str, Callable[[Any], bool]]): 一个字典，键是参数名，值是一个校验函数。
            校验函数接收参数值，如果通过校验则返回True，否则返回False。
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # 将位置参数和关键字参数转换为一个字典，便于统一处理
            # 这部分代码稍微复杂，因为需要正确关联参数名和值
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()  # 应用函数定义中的默认参数值
            original_query = bound_args.arguments.get('original_query')  # 获取用户原始问题
            cur_func_name = func.__name__
            # 公共校验逻辑--工具调用次数限制
            if bound_args.arguments.get('is_exceed'):
                return ToolResponse(content=[TextBlock(type='text',
                                                       text="工具调用次数已达上限，请基于已获取的信息总结工具调用结果后传递给运维专家Agent，不可继续调用工具。若信息不足，可告知用户当前能提供的相关内容。")])

            # 遍历所有校验规则
            for param_name, validator in validators.items():
                # 检查函数是否有这个参数
                if param_name not in sig.parameters:
                    raise ValueError(f"校验规则中包含了函数 '{func.__name__}' 没有的参数 '{param_name}'")
                # 获取参数的值
                param_value = bound_args.arguments.get(param_name)
                # 执行个性化校验
                if not validator(param_value):
                    # 校验失败，返回默认回复
                    print(f"方法{func.__name__}的参数校验失败: 参数 '{param_name}' 的值 '{param_value}' 未通过校验。")
                    return ToolResponse(content=[TextBlock(type='text',
                                                           text=f"工具调用异常：{cur_func_name}工具的{param_name}参数为空。请基于用户输入「{original_query}」和当前已经得到的工具调用结果，生成具体检索关键词")])
            # 所有参数都通过校验，执行原函数
            return func(*args, **kwargs)

        return wrapper

    return decorator


# -------------------------- 1. 全局配置与无状态工具/知识库（共享） --------------------------
# 工具1：运维手册知识库检索工具（基于FAISS轻量向量库）
class MaintenanceDocRetriever:
    """运维手册知识库检索工具，支持设备故障、操作规范查询"""

    def __init__(self, index_path: str = "maintenance_index.index", texts_path: str = "maintenance_texts.pkl"):
        # 加载向量化模型和FAISS索引
        self.embedder = SentenceTransformer(r'C:\Users\gaohu\aiModel\text2vec-base-chinese')
        self.index = faiss.read_index(index_path) if os.path.exists(index_path) else self._init_index()
        self.texts = self._load_texts(texts_path) if os.path.exists(texts_path) else self._init_knowledge()

    def _init_index(self) -> faiss.Index:
        """初始化FAISS索引"""
        dim = 768  # text2vec-base-chinese 的维度
        index = faiss.IndexFlatL2(dim)  # 使用简单的L2距离索引
        return index

    def _init_knowledge(self) -> List[str]:
        """初始化运维手册知识库（实际场景可从PDF/Word导入）"""
        knowledge = [
            # 设备故障排查
            "设备型号M-2000报警代码E101：冷却系统压力不足，检查冷却液液位和水泵是否正常",
            "设备型号M-2000报警代码E203：主轴转速异常，排查电机接线和变频器参数",
            "设备型号M-3000报警代码E056：液压系统泄漏，检查密封圈和油管接口",
            # 操作规范
            "M系列设备开机前必须检查：电源电压（380V±10%）、润滑油液位、安全防护门闭合状态",
            "设备急停按钮触发后，需先排查故障原因，再按复位键重启，禁止直接通电",
            # 保养周期
            "主轴轴承保养周期：每运行2000小时更换润滑脂",
            "冷却系统滤芯更换周期：每3个月或运行1500小时",
            "液压油更换周期：每年一次，或运行5000小时",
            # 设备参数
            "M-2000最大加工转速：8000rpm，额定负载：500kg",
            "M-3000最大加工转速：10000rpm，额定负载：800kg"
        ]
        # 向量化并构建索引
        vectors = self.embedder.encode(knowledge, convert_to_numpy=True).astype('float32')
        # if not self.index.is_trained:
        #     self.index.train(vectors)
        self.index.add(vectors)
        # 保存索引和文本
        faiss.write_index(self.index, "maintenance_index.index")
        with open("maintenance_texts.pkl", "wb") as f:
            pickle.dump(knowledge, f)
        return knowledge

    def _load_texts(self, path: str) -> List[str]:
        """加载知识库文本"""
        with open(path, "rb") as f:
            return pickle.load(f)

    @validate_params(
        validators={
            # 校验 'query' 参数必须是字符串且非空
            "query": lambda q: isinstance(q, str) and q.strip() != ""
        }
    )
    def kb_search(self, query: str = None, top_k: int = 3, is_exceed: bool = False,
                  original_query: str = None) -> ToolResponse:
        """
        运维手册知识库检索工具，支持设备故障、操作规范查询
        Args:
            query: 用户输入与运维场景有关的问题
            top_k: 检索数据时取前top_k个与用户输入最相关的分块
        Returns:
            与用户输入相关的分块内容列表
        """
        """检索相关知识库内容"""
        query_vec = self.embedder.encode([query], convert_to_numpy=True).astype('float32')
        distances, indices = self.index.search(query_vec, top_k)
        results = []
        for i in indices[0]:
            if i < len(self.texts):
                results.append(TextBlock(type='text', text=self.texts[i]))
        # return results
        return ToolResponse(content=results)


# 初始化工具实例
doc_retriever = MaintenanceDocRetriever()


@validate_params(
    validators={
        # 校验 'device_model' 参数必须是字符串且非空
        "device_model": lambda q: isinstance(q, str) and q.strip() != "",
        # 校验 'error_code' 参数必须是字符串且非空
        "error_code": lambda q: isinstance(q, str) and q.strip() != ""
    }
)
# 工具2：故障代码解析工具（结构化输出）
def parse_error_code(device_model: str, error_code: str, is_exceed: bool = False,
                     original_query: str = None) -> ToolResponse:
    """
    解析设备故障代码的详细信息
    Args:
        device_model: 设备型号（如M-2000、M-3000）
        error_code: 故障代码（如E101、E203）
    Returns:
        故障解析结果（包含原因、排查步骤、解决方案）
    """
    error_db = {
        "M-2000": {
            "E101": {
                "原因": "冷却系统压力不足",
                "排查步骤": "1. 检查冷却液液位是否低于警戒线；2. 启动水泵，听是否有异响；3. 检查冷却管路是否堵塞",
                "解决方案": "补充冷却液至标准液位；清理管路堵塞物；水泵故障则更换水泵"
            },
            "E203": {
                "原因": "主轴转速异常",
                "排查步骤": "1. 检查电机接线是否松动；2. 查看变频器显示参数；3. 测试主轴轴承是否卡滞",
                "解决方案": "紧固电机接线；重新校准变频器参数；轴承卡滞则更换轴承"
            }
        },
        "M-3000": {
            "E056": {
                "原因": "液压系统泄漏",
                "排查步骤": "1. 检查液压油管接口是否松动；2. 查看密封圈是否老化；3. 检测液压泵压力",
                "解决方案": "紧固接口螺栓；更换老化密封圈；液压泵压力不足则维修或更换"
            }
        }
    }
    if device_model in error_db and error_code in error_db[device_model]:
        target_info = error_db[device_model][error_code]
        # 将字典转换为格式化的文本字符串
        content = f"故障代码解析结果：\n设备型号: {device_model}\n报警代码: {error_code}\n"
        content += f"故障原因: {target_info['原因']}\n"
        content += f"排查步骤: {target_info['排查步骤']}\n"
        content += f"解决方案: {target_info['解决方案']}"
        result = TextBlock(type='text', text=content)
    else:
        result = TextBlock(type='text',
                           text=f"未知故障代码，排查步骤：无该{device_model}型号的{error_code}代码记录，请基于用户问题查询运维手册")
    return ToolResponse(content=[result])


@validate_params(
    validators={
        # 校验 'part_name' 参数必须是字符串且非空
        "part_name": lambda q: isinstance(q, str) and q.strip() != ""
    }
)
# 工具3：保养周期查询工具
def query_maintenance_cycle(part_name: str, is_exceed: bool = False, original_query: str = None) -> ToolResponse:
    """
    查询设备部件的保养周期
    Args:
        part_name: 部件名称（如主轴轴承、冷却系统滤芯、液压油）
    Returns:
        保养周期说明
    """
    cycle_db = {
        "主轴轴承": "每运行2000小时更换润滑脂",
        "冷却系统滤芯": "每3个月或运行1500小时更换",
        "液压油": "每年一次，或运行5000小时更换",
        "安全防护门": "每月检查一次闭合灵敏度",
        "变频器": "每6个月校准一次参数"
    }
    return ToolResponse(
        content=[TextBlock(type='text', text=cycle_db.get(part_name, f"无{part_name}的保养周期记录，请参考运维手册"))])


# -------------------------- 2. Agent实例状态枚举 --------------------------
class AgentInstanceState(Enum):
    IDLE = "空闲"  # 可被复用
    BUSY = "忙碌"  # 已绑定会话，正在处理


# -------------------------- 3. 单个Agent实例封装（含检索+专家对） --------------------------
# 定义全局的 pre_acting 钩子（用于所有检索助手实例）


def create_pre_acting_hook(max_tool_calls: int = MAX_TOOL_CALLS_PER_DIALOG):
    """
    创建实例级 pre_acting 钩子（带独立计数器，避免多实例冲突）
    - max_tool_calls: 该 Agent 允许的最大工具调用次数

    【除了预处理钩子外，还有后处理钩子，用于 “修改 / 拦截工具调用结果”】
    TODO 注意由于ReActAgent实例的_acting方法仅当调用结束工具时才会返回非空工具结果【其他普通工具，该方法皆返回None】，因此后处理钩子不便用于处理普通工具的结果后处理
    """

    async def pre_acting_hook(agent: ReActAgent, kwargs: Dict[str, Any]) -> Optional[
        Dict[str, Any]]:  # 预处理钩子的设计目标是 “修改 / 拦截工具调用参数”。
        cur_tool_name = kwargs.get("tool_call", {}).get("name")
        if cur_tool_name == "generate_response":  # agentscope框架中的结束工具方法
            # 从智能体实例缓存中获取当前正在处理的对话id【对应一次对话交互】
            user_msg_list = [msg for msg in agent.memory.content if msg.role == "user" and msg.invocation_id]
            cur_user_msg = user_msg_list[-1]  # 取最近的用户消息作为当前用户消息
            input_param = kwargs.get("tool_call", {}).get("input", {})
            if not input_param or input_param.get("response") in (None, ""):
                kwargs.get("tool_call", {}).get("input", {}).setdefault("response",
                                                                        f"针对【{cur_user_msg.content}】无任何有效检索结果，请将该情况告知运维专家agent")
        else:
            # 从智能体实例缓存中获取当前正在处理的对话id【对应一次对话交互】
            user_msg_list = [msg for msg in agent.memory.content if msg.role == "user" and msg.invocation_id]
            cur_user_msg = user_msg_list[-1]  # 取最近的用户消息作为当前用户消息
            cur_conversation_id = cur_user_msg.invocation_id
            kwargs.get("tool_call", {}).get("input", {}).setdefault("original_query",
                                                                    cur_user_msg.content)  # 将用户原始问题传入工具，以备异常情况提示
            # 1. 从缓存获取当前调用次数（不存在则视为0，避免初始化遗漏）
            current_count = TOOL_CALL_CACHE.get(cur_conversation_id, 1)
            if current_count > max_tool_calls:
                logger.info(f"Agent {agent.id} 工具调用达上限，停止调用并提示模型总结回答")
                # 关键：向工具调用参数中添加调用次数达到最大次数的标识，用以指示工具返回默认回复“工具调用次数已达上限（最多3次），请基于已获取的信息直接生成最终回答，无需继续调用工具。若信息不足，可告知用户当前能提供的相关内容。”
                kwargs.get("tool_call", {}).get("input", {}).setdefault("is_exceed", True)  # 提示工具当前已超过调用次数限制

            logger.info(f"Agent {agent.name}（{agent.id}）（{cur_conversation_id}）第 {current_count} 次调用工具：{kwargs}")
            TOOL_CALL_CACHE[cur_conversation_id] = current_count + 1  # 为下一次工具调用追加1
        return kwargs  # 返回 None → AgentScope 会终止本次工具调用

    return pre_acting_hook


class AgentPair:
    """Agent实例对：包含一个检索Agent和一个专家Agent，绑定同一会话"""

    def __init__(self, pair_id: str):
        self.pair_id = pair_id  # 实例对ID
        self.session_id: Optional[str] = None  # 绑定的会话ID（空闲时为None）
        self.message_id: Optional[str] = None  # 绑定的对话消息ID（空闲时为None）
        self.model = DashScopeChatModel(
            model_name="deepseek-v3",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
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

    def _create_retriever_agent(self) -> ReActAgent:
        # 初始化工具箱实例
        toolkit = Toolkit()
        # 使用 register_tool_function 方法注册工具
        toolkit.register_tool_function(doc_retriever.kb_search)
        toolkit.register_tool_function(parse_error_code)
        toolkit.register_tool_function(query_maintenance_cycle)

        retriever = ReActAgent(
            name="检索助手",
            sys_prompt="""仅处理当前绑定会话的用户问题：
1. 基于专属内存中的对话历史调用工具，结果整理后传递给运维专家Agent；
2. 不直接回复用户，仅输出工具调用结果；
3. 会话切换时内存会清空，无需考虑历史会话信息。
4. 若收到“工具调用异常+强制终止”的系统提示，立即停止所有工具调用，基于现有信息总结工具调用结果后传递给运维专家Agent；
5. 必须将输出结果控制在100字以内。""",
            model=self.model,
            formatter=DashScopeMultiAgentFormatter(),
            toolkit=toolkit
        )
        retriever.set_console_output_enabled(False)  # 禁用控制台输出智能体内容，由业务代码控制输出内容
        # 为当前 Agent 实例注册 预处理钩子
        retriever.register_instance_hook(
            hook_type="pre_acting",  # 钩子类型：工具调用前
            hook_name="tool_use_check",  # 钩子名称（唯一标识）
            hook=create_pre_acting_hook(max_tool_calls=3)  # 传入带工具调用最大次数的钩子
        )
        return retriever

    def _create_expert_agent(self) -> ReActAgent:
        expert = ReActAgent(
            name="运维专家",
            sys_prompt="""仅处理当前绑定会话的用户问题：
1. 基于专属内存中的用户问题和检索结果，按"原因→步骤→解决方案→注意事项"组织回复；
2. 仅关注当前会话信息，会话切换时内存会清空；
3. 多轮对话记住已提供内容，避免重复。""",
            model=self.model,
            formatter=DashScopeMultiAgentFormatter()
        )
        expert.set_console_output_enabled(False)  # 禁用控制台输出智能体内容，由业务代码控制输出内容
        return expert

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
            if self.session_id in SESSION_GOING_CACHE:
                del SESSION_GOING_CACHE[self.session_id]
            self.session_id = None
            logger.info(f"Agent实例对 {self.pair_id} 解除会话绑定，回归空闲")

    def get_state(self) -> AgentInstanceState:
        """获取当前实例状态"""
        if self.session_id is None:
            return AgentInstanceState.IDLE
        return AgentInstanceState.BUSY


# -------------------------- 4. Agent实例池（核心：动态调度+复用） --------------------------
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
            # if idle_pairs:
            #     agent_pair = idle_pairs[0]
            #     await agent_pair.bind_session(session_id)
            #     return agent_pair
            # 新代码如下:
            if idle_pairs:
                for agent_pair in idle_pairs:
                    async with agent_pair.instance_lock:  # 加锁验证，避免已被绑定
                        if agent_pair.get_state() != AgentInstanceState.IDLE:
                            continue  # 实例已被其他协程绑定，重新查找
                        await agent_pair.bind_session(session_id)
                    return agent_pair

            # 3. 无空闲实例，且未达最大容量→扩容
            if len(self.pool) < self.max_size:
                new_pair = AgentPair(pair_id=f"agent_pair_{len(self.pool) + 1}")
                # 这里是先绑定会话,然后再加入智能体实例池,属于同步操作,无需加协程锁校验
                await new_pair.bind_session(session_id)
                self.pool.append(new_pair)
                logger.info(f"实例池扩容，当前容量：{len(self.pool)}")
                return new_pair

            # 4. 已达最大容量→等待空闲实例（超时1分钟）
            logger.warning("实例池已达最大容量，用户会话排队中...")
            return await self._wait_for_idle_pair(session_id, timeout=120)  # 排队时长增加到2分钟

    async def _wait_for_idle_pair(self, session_id: str, timeout: int = 60) -> AgentPair:
        """等待空闲实例（超时抛出异常）"""
        start_time = datetime.now()
        while datetime.now() - start_time < timedelta(seconds=timeout):
            await asyncio.sleep(5)  # 每5秒检查一次
            async with self.lock:
                idle_pairs = [p for p in self.pool if p.get_state() == AgentInstanceState.IDLE]
                if idle_pairs:
                    agent_pair = idle_pairs[0]
                    await agent_pair.bind_session(session_id)
                    logger.info(f"会话 {session_id} 成功获取空闲实例")
                    return agent_pair
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


# -------------------------- 5. 会话管理与多用户服务 --------------------------
class SessionManager:
    @staticmethod
    def create_session() -> str:
        """创建新会话ID"""
        return f"session_{uuid.uuid4().hex[:8]}"


async def user_dialog_for_one_question(session_id: str, agent_pool: AgentPool, question: str, conversation_id: str):
    """单个用户多轮对话（复用Agent实例）"""
    logger.info(f"\n【会话 {session_id}】用户开始咨询")
    try:
        # 1. 获取绑定该会话的Agent实例对
        agent_pair = await agent_pool.get_agent_pair(session_id)
        retriever = agent_pair.retriever
        expert = agent_pair.expert
        # 2. 构造用户消息
        user_msg = Msg(name="user", content=question, role="user", invocation_id=conversation_id)

        # 3. 存入实例专属内存
        await retriever.memory.add(user_msg)
        await expert.memory.add(user_msg)
        logger.info(f"【会话 {session_id}】发送：{question}")

        # 4. 智能体协作处理
        retriever_reply = await retriever.reply()
        logger.info(f"【会话 {session_id}】的检索助手返回数据：{retriever_reply.content}")
        await expert.memory.add(retriever_reply)
        expert_reply = await expert.reply()
        # TODO 在得到运维专家回复后，清除智能体实例中的缓存数据【历史对话及工作调用数据】，并将当前会话与当前智能体实例接触绑定，后续新对话进来重新从资源池获取新的智能体实例
        await agent_pair.unbind_session()
        # 5. 输出结果
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
agent_pool = AgentPool(min_size=20, max_size=20)


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
        result = await user_dialog_for_one_question(session_id, agent_pool, request.question, conversation_id)
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
    # asyncio.run(main())
    import uvicorn

    uvicorn.run(app='demo:app', port=8090, reload=False,
                workers=1)
