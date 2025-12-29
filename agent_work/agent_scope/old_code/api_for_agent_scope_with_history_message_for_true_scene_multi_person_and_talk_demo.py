"""
某大型工业设备公司（如机床、发电机组厂商）需构建运维问答 Agent，支持：
    基于运维手册知识库回答设备故障排查、操作规范、保养周期等问题；
    多智能体协作（检索 Agent 负责知识库查询，专家 Agent 负责逻辑推理）；
    工具调用（文档检索、故障代码解析工具）；
    上下文记忆（多轮对话中记住设备型号、历史故障等信息）。
    多人多轮对话功能：基于智能体池
"""
import random
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from enum import Enum

"""
本地安装的agentscope版本为1.0.7，AgentScope 1.0.7 版本的关键特点
    Tools 使用方式：直接传递函数列表给 tools 参数
    
    不需要特殊装饰器：普通 Python 函数即可作为工具，具体使用如下：
        第一步：初始化工具箱实例，并将python函数注册到工具箱中
            # 正确的初始化方式
            toolkit = Toolkit()
            # 使用 register_tool_function 方法注册工具
            toolkit.register_tool_function(doc_retriever.kb_search)
            toolkit.register_tool_function(parse_error_code)
            toolkit.register_tool_function(query_maintenance_cycle)
        第二步：
            **_agent = ReActAgent(
                name="智能体名称",
                sys_prompt="系统提示词说明",
                model=model,  # 大模型实例
                formatter=DashScopeMultiAgentFormatter(),  # 消息格式
                # 在 AgentScope 1.0.7 版本中，ReActAgent 使用的是 toolkit 实例【在初始化后，需要将工具手动注册到该实例中】
                toolkit=toolkit
            )
    
    MsgHub 使用：使用 hub.broadcast() 方法发送消息
    
    异步处理：使用 await agent() 调用智能体
"""
"""    
    AgentScope 1.0.7 中多智能体通信的完整逻辑—— 核心是「** 广播接收 + 记忆过滤 + 定向回复 **」，完全适配框架的轻量协作设计。以下是具体拆解、通信流程和定向协作实现：

        1. ReActAgent实例的observe 方法：被动接收广播，存入记忆（无过滤）
        核心作用：接收 MsgHub 广播的所有消息，直接存入智能体的 memory（记忆模块），** 不做任何过滤 **；
        关键细节：
            支持单个 / 多个 Msg 输入，都会被 memory.add(msg) 批量存入；
            无返回值，仅负责 “接收并记录”，不触发智能体生成回复；
            所有注册到 MsgHub 的 ReActAgent 都会收到广播消息，且全部存入自身记忆。
        2. reply 方法：主动生成回复，基于记忆筛选有效信息
        核心作用：根据输入消息（可选）和自身记忆，生成回复 Msg，实现 “定向协作” 的关键在「记忆筛选」；
        关键细节：
            输入 msg 可选：若传入，会优先基于该消息触发思考；若未传入，会从 memory 中读取历史消息；
            生成的回复是 Msg 对象：name 为当前智能体名称，role 默认为 assistant，content 为思考结果；
            核心逻辑：ReActAgent 会在 reply 中自动筛选记忆里的 “相关消息”（如来自目标发送者、特定角色的消息），忽略无关信息。自动筛选相关信息的依据在于当前智能体的sys_prompt设置。
    
    
    1. 检索 Agent 的 sys_prompt（只处理用户消息）
        python代码
        retriever = ReActAgent(
            name="检索助手",
            sys_prompt=
            1. 只处理来自"user"的消息（用户的咨询问题），忽略其他发送者的消息；
            2. 根据用户问题调用工具（故障解析/知识库检索），生成工具结果；
            3. 回复内容仅包含工具结果，不直接给用户解决方案。,
            tools=[...],
            model=...
        )
    2. 专家 Agent 的 sys_prompt（只处理检索 Agent 消息）
        python代码
        expert = ReActAgent(
            name="运维专家",
            sys_prompt=
            1. 只处理来自"检索助手"的消息（工具结果），忽略其他发送者的消息；
            2. 结合用户原始问题（从记忆中读取）和工具结果，生成解决方案；
            3. 回复内容按"原因→步骤→解决方案→注意事项"组织，面向用户。,
            model=...
        )
    效果：
        用户消息广播后，只有检索 Agent 会处理并生成工具结果；
        检索 Agent 的回复广播后，只有专家 Agent 会处理并生成最终答案；
        全程无 receiver，但通过 “发送者名称 + 系统提示词筛选” 实现了精准定向协作。
    
    
    设计优势与适用场景
        优势：
        简化开发：无需手动指定 receiver 和消息路由，通过 sys_prompt 即可定义协作规则，快速搭建多智能体流程；
        记忆自动共享：所有消息存入智能体记忆，支持多轮对话上下文复用（如专家 Agent 能读取用户原始问题和检索结果）；
        灵活性高：筛选规则可通过 sys_prompt 动态调整（如后续新增 “审核 Agent”，只需修改规则让其处理专家 Agent 的回复）。
    适用场景：
        中小规模协作（2-5 个智能体）：如 “检索→推理→回复” 的简单流程；
        流程固定的场景：智能体分工明确（如固定 “用户→工具调用 Agent→专家 Agent”），无需复杂路由；
        快速原型验证：无需关注通信细节，聚焦智能体的核心逻辑（工具调用、推理）。
"""
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
                                                       text="工具调用次数已达上限，请基于已获取的信息总结工具调用结果后传递给运维专家Agent，不可继续调用工具。若信息不足，可告知用户当前能提供的相关内容。")], metadata={"mark": "invalid_msg"})

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
                    return ToolResponse(content=[TextBlock(type='text', text=f"工具调用异常：{cur_func_name}工具的{param_name}参数为空。请基于用户输入「{original_query}」和当前已经得到的工具调用结果，生成具体检索关键词")], metadata={"mark": "invalid_msg"})
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
        # self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedder = SentenceTransformer(r'C:\Users\gaohu\aiModel\text2vec-base-chinese')
        self.index = faiss.read_index(index_path) if os.path.exists(index_path) else self._init_index()
        self.texts = self._load_texts(texts_path) if os.path.exists(texts_path) else self._init_knowledge()

    def _init_index(self) -> faiss.Index:
        """初始化FAISS索引"""
        # dim = 768
        # index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, 100, faiss.METRIC_L2)
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
        with open("../agent/maintenance_texts.pkl", "wb") as f:
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
    def kb_search(self, query: str = None, top_k: int = 3, is_exceed: bool = False, original_query: str = None) -> ToolResponse:
        """
        运维手册知识库检索工具，支持设备故障、操作规范查询
        Args:
            query: 用户输入与运维场景有关的问题
            top_k: 检索数据时取前top_k个与用户输入最相关的分块
        Returns:
            与用户输入相关的分块内容列表
        """
        # # == start-在工具中添加参数校验机制，避免模型调用时因确实入参而报错，同时提醒模型需传入参数 ==
        # # 参数校验：拒绝空字符串、纯空格
        # if not query or query.strip() == "":
        #     return ToolResponse(content=[TextBlock(type='text', text="Error: 检索关键词不能为空，请提供具体查询内容")])
        # # == end ==
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
def parse_error_code(device_model: str, error_code: str, is_exceed: bool = False, original_query: str = None) -> ToolResponse:
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
        # result = {"原因": "未知故障代码", "排查步骤": f"无该{device_model}型号的{error_code}代码记录",
        #         "解决方案": "联系技术支持"}
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
    EXPIRED = "过期"  # 超时未使用，待回收


# -------------------------- 3. 单个Agent实例封装（含检索+专家对） --------------------------
# 定义全局的 pre_acting 钩子（用于所有检索助手实例）


def create_pre_acting_hook(max_tool_calls: int = MAX_TOOL_CALLS_PER_DIALOG):
    """
    创建实例级 pre_acting 钩子（带独立计数器，避免多实例冲突）
    - max_tool_calls: 该 Agent 允许的最大工具调用次数

    【除了预处理钩子外，还有后处理钩子，用于 “修改 / 拦截工具调用结果”】
    TODO 注意由于ReActAgent实例的_acting方法仅当调用结束工具时才会返回非空工具结果【其他普通工具，该方法皆返回None】，因此后处理钩子不便用于处理普通工具的结果后处理
    """

    async def pre_acting_hook(agent: ReActAgent, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:  # 预处理钩子的设计目标是 “修改 / 拦截工具调用参数”。
        cur_tool_name = kwargs.get("tool_call", {}).get("name")
        if cur_tool_name == "generate_response":  # agentscope框架中的结束工具方法
            # 清除检索助手中系统追加工具调用异常数据缓存，避免影响后续对话【一些模型可能会将历史对话中的工具调用异常提示作用在后续新会话上】
            invalid_msg_list = [msg for msg in agent.memory.content if msg.metadata and msg.metadata.get("mark", "") == "invalid_msg"]
            if invalid_msg_list:
                invalid_tool_use_ids = [msg.metadata.get("tool_use_id") for msg in invalid_msg_list]
                valid_msg_list = []
                for msg in agent.memory.content:
                    cur_content = msg.content
                    if msg.metadata and msg.metadata.get("mark", "") == "invalid_msg":  # 过滤掉自定义异常工具调用结果
                        continue
                    if isinstance(cur_content, list) and cur_content[0]['id'] in invalid_tool_use_ids:
                        continue  # 无效工具调用请求，过滤掉
                    valid_msg_list.append(msg)
                await agent.memory.clear()
                agent.memory.content = valid_msg_list
            # 从智能体实例缓存中获取当前正在处理的对话id【对应一次对话交互】
            user_msg_list = [msg for msg in agent.memory.content if msg.role == "user" and msg.invocation_id]
            cur_user_msg = user_msg_list[-1]  # 取最近的用户消息作为当前用户消息
            input_param = kwargs.get("tool_call", {}).get("input", {})
            if not input_param or input_param.get("response") in (None, ""):
                kwargs.get("tool_call", {}).get("input", {}).setdefault("response", f"针对【{cur_user_msg.content}】无任何有效检索结果，请将该情况告知运维专家agent")
        else:
            # 从智能体实例缓存中获取当前正在处理的对话id【对应一次对话交互】
            user_msg_list = [msg for msg in agent.memory.content if msg.role == "user" and msg.invocation_id]
            cur_user_msg = user_msg_list[-1]  # 取最近的用户消息作为当前用户消息
            cur_conversation_id = cur_user_msg.invocation_id
            kwargs.get("tool_call", {}).get("input", {}).setdefault("original_query", cur_user_msg.content)  # 将用户原始问题传入工具，以备异常情况提示
            # 1. 从缓存获取当前调用次数（不存在则视为0，避免初始化遗漏）
            current_count = TOOL_CALL_CACHE.get(cur_conversation_id, 1)
            if current_count > max_tool_calls:
                # # 关键：向 Agent 内存添加系统提示，告知停止工具调用 TODO 看起来不起作用
                # stop_msg = Msg(
                #     name="system",
                #     content="工具调用次数已达上限（最多3次），请基于已获取的信息直接生成最终回答，无需继续调用工具。若信息不足，可告知用户当前能提供的相关内容。",
                #     role="system"
                # )
                # agent.memory.content.append(stop_msg)  # 将提示加入 Agent 记忆
                logger.info(f"Agent {agent.id} 工具调用达上限，停止调用并提示模型总结回答")
                # 关键：向工具调用参数中添加调用次数达到最大次数的标识，用以指示工具返回默认回复“工具调用次数已达上限（最多3次），请基于已获取的信息直接生成最终回答，无需继续调用工具。若信息不足，可告知用户当前能提供的相关内容。”
                kwargs.get("tool_call", {}).get("input", {}).setdefault("is_exceed", True)  # 提示工具当前已超过调用次数限制

            logger.info(f"Agent {agent.name}（{agent.id}）（{cur_conversation_id}）第 {current_count} 次调用工具：{kwargs}")
            TOOL_CALL_CACHE[cur_conversation_id] = current_count + 1  # 为下一次工具调用追加1
        return kwargs  # 返回 None → AgentScope 会终止本次工具调用

    return pre_acting_hook


class ReActAgentSelf(ReActAgent):

    async def _acting(self, tool_call: ToolUseBlock) -> Msg | None:
        """Perform the acting process.

        Args:
            tool_call (`ToolUseBlock`):
                The tool use block to be executed.

        Returns:
            `Union[Msg, None]`:
                Return a message to the user if the `finish_function` is
                called, otherwise return `None`.
        """

        tool_res_msg = Msg(
            "system",
            [
                ToolResultBlock(
                    type="tool_result",
                    id=tool_call["id"],
                    name=tool_call["name"],
                    output=[],
                ),
            ],
            "system",
        )
        try:
            # Execute the tool call
            tool_res = await self.toolkit.call_tool_function(tool_call)

            response_msg = None
            # Async generator handling
            async for chunk in tool_res:
                # Turn into a tool result block
                tool_res_msg.content[0][  # type: ignore[index]
                    "output"
                ] = chunk.content
                # 定制化：向缓存消息中添加工具装饰器中的异常消息标识，用于在调用结束工具前将这些异常消息过滤剔除，避免影响模型对后续对话的正常处理
                if chunk.metadata and chunk.metadata.get("mark"):
                    tool_res_msg.metadata = chunk.metadata  # 将自定义的元数据添加到工具调用结果中
                    tool_res_msg.metadata["tool_use_id"] = tool_call["id"]  # 将工具调用id一并追加到工具调用结果中，用于删除异常结果时一并删除相应的工具调用请求，以避免后续模型reason时因工具调用请求与结果不对等而报错

                # Skip the printing of the finish function call
                if (
                    tool_call["name"] != self.finish_function_name
                    or tool_call["name"] == self.finish_function_name
                    and (
                        chunk.metadata is None
                        or not chunk.metadata.get("success")
                    )
                ):
                    await self.print(tool_res_msg, chunk.is_last)

                # Raise the CancelledError to handle the interruption in the
                # handle_interrupt function
                if chunk.is_interrupted:
                    raise asyncio.CancelledError()

                # Return message if generate_response is called successfully
                if (
                    tool_call["name"] == self.finish_function_name
                    and chunk.metadata
                    and chunk.metadata.get(
                        "success",
                        True,
                    )
                ):
                    response_msg = chunk.metadata.get("response_msg")

            return response_msg

        finally:
            # Record the tool result message in the memory
            await self.memory.add(tool_res_msg)


class AgentPair:
    """Agent实例对：包含一个检索Agent和一个专家Agent，绑定同一会话"""

    def __init__(self, pair_id: str):
        self.pair_id = pair_id  # 实例对ID
        self.session_id: Optional[str] = None  # 绑定的会话ID（空闲时为None）
        self.message_id: Optional[str] = None  # 绑定的对话消息ID（空闲时为None）
        self.bind_time: Optional[datetime] = None  # 绑定会话的时间
        self.idle_timeout = timedelta(minutes=10)  # 空闲超时时间（10分钟无操作则释放）
        self.model = DashScopeChatModel(
            model_name="qwen3-max",
            api_key="sk-6b8afa231399490bb7a56c025a3bc633",
            # api_key=os.getenv("DASHSCOPE_API_KEY"),
            # temperature=0.1  # 降低随机性，保证运维回答准确性
            generate_kwargs={
                "temperature": 0.1,
                "top_p": 0.8,
                "max_tokens": 300,
                "repetition_penalty": 1.1
            }
        )

        # 创建检索Agent和专家Agent
        self.retriever = self._create_retriever_agent()
        self.expert = self._create_expert_agent()

    def _create_retriever_agent(self) -> ReActAgentSelf:
        # == start-添加工具调用次数限制 ==
        # == end ==
        # 初始化工具箱实例
        toolkit = Toolkit()
        # 使用 register_tool_function 方法注册工具
        toolkit.register_tool_function(doc_retriever.kb_search)
        toolkit.register_tool_function(parse_error_code)
        toolkit.register_tool_function(query_maintenance_cycle)

        retriever = ReActAgentSelf(
            name="检索助手",
            sys_prompt="""仅处理当前绑定会话的用户问题：
1. 基于专属内存中的对话历史调用工具，结果整理后传递给运维专家Agent；
2. 不直接回复用户，仅输出工具调用结果；
3. 会话切换时内存会清空，无需考虑历史会话信息。
4. 若收到“工具调用异常+强制终止”的系统提示，立即停止所有工具调用，基于现有信息总结工具调用结果后传递给运维专家Agent；
5. 必须将输出结果控制在100字以内。""",
            model=self.model,
            # memory=self.retriever_memory,
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
            # memory=self.expert_memory,
            formatter=DashScopeMultiAgentFormatter()
        )
        expert.set_console_output_enabled(False)  # 禁用控制台输出智能体内容，由业务代码控制输出内容
        return expert

    async def bind_session(self, session_id: str) -> None:
        """绑定会话：清理内存+记录会话信息"""
        # 清理上一会话的内存（核心：保证隔离）
        await self.retriever.memory.clear()
        await self.expert.memory.clear()
        # 绑定新会话
        self.session_id = session_id
        self.bind_time = datetime.now()
        logger.info(f"Agent实例对 {self.pair_id} 绑定会话 {session_id}")

    async def unbind_session(self) -> None:
        """解除会话绑定：清理内存+恢复空闲状态"""
        await self.retriever.memory.clear()
        await self.expert.memory.clear()
        # 将当前会话从会话是否进行中的缓存中清除
        if self.session_id in SESSION_GOING_CACHE:
            del SESSION_GOING_CACHE[self.session_id]
        self.session_id = None
        self.bind_time = None
        logger.info(f"Agent实例对 {self.pair_id} 解除会话绑定，回归空闲")

    def get_state(self) -> AgentInstanceState:
        """获取当前实例状态"""
        if self.session_id is None:
            return AgentInstanceState.IDLE
        # 检查会话是否超时（绑定后10分钟无操作）
        if datetime.now() - self.bind_time > self.idle_timeout:
            return AgentInstanceState.EXPIRED
        return AgentInstanceState.BUSY


# -------------------------- 4. Agent实例池（核心：动态调度+复用） --------------------------
class AgentPool:
    def __init__(self, min_size: int = 5, max_size: int = 20):
        self.min_size = min_size  # 实例池最小容量（预创建）
        self.max_size = max_size  # 实例池最大容量（扩容上限）
        self.pool: List[AgentPair] = []  # 实例池存储
        self.lock = asyncio.Lock()  # 并发安全锁
        self.session_map: Dict[str, AgentPair] = {}  # 会话→实例对映射，用以记录当前已经在用的智能体实例对【会话id：智能体实例对】

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
            # 1. 检查会话是否已绑定实例（多轮对话复用同一实例）
            if session_id in self.session_map:
                agent_pair = self.session_map[session_id]
                # 检查实例状态：若过期则重新绑定
                if agent_pair.get_state() == AgentInstanceState.EXPIRED:
                    await agent_pair.bind_session(session_id)
                return agent_pair

            # 2. 查找空闲实例
            idle_pairs = [p for p in self.pool if p.get_state() == AgentInstanceState.IDLE]
            if idle_pairs:
                agent_pair = idle_pairs[0]
                await agent_pair.bind_session(session_id)
                self.session_map[session_id] = agent_pair
                return agent_pair

            # 3. 无空闲实例，且未达最大容量→扩容
            if len(self.pool) < self.max_size:
                new_pair = AgentPair(pair_id=f"agent_pair_{len(self.pool) + 1}")  # 扩容时，因创建智能体实例较慢，导致中断请求连接
                await new_pair.bind_session(session_id)
                self.pool.append(new_pair)
                self.session_map[session_id] = new_pair
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
                    self.session_map[session_id] = agent_pair
                    logger.info(f"会话 {session_id} 成功获取空闲实例")
                    return agent_pair
        raise TimeoutError("当前咨询用户过多，请稍后重试")

    async def clean_expired_pairs(self):
        """
        当前api服务重载时，此处会报错“asyncio.exceptions.CancelledError”。本质是「执行时机+异步任务取消机制」导致了报错——核心原因是：**服务重载时后台任务被强制取消，而 `asyncio.sleep` 是可取消的异步操作，取消时会直接抛出 `CancelledError`**。

        ### 先理清关键逻辑：为什么是 `asyncio.sleep` 触发报错？
        我们把整个过程拆成3步，就能看懂因果：
        1. **后台任务的运行状态**：你的 `clean_expired_pairs` 是个「无限循环+休眠」的异步任务，逻辑是：
           ```python
           async def clean_expired_pairs(self):
               while True:  # 无限循环
                   执行清理逻辑...
                   await asyncio.sleep(60)  # 休眠60秒（等待下一次清理）
           ```
           大多数时间里，这个任务都处于 `asyncio.sleep(60)` 的「休眠等待状态」（而非执行计算）。

        2. **服务重载触发任务取消**：当 FastAPI 检测到文件变更，触发自动重载时，会执行「关闭旧服务」流程：
           - 你的 `lifespan` 关闭逻辑里调用了 `clean_task.cancel()`（主动取消后台任务）；
           - 异步任务的「取消机制」是：如果任务正在执行「可取消的异步操作」（比如 `sleep`、网络请求等），会立刻中断该操作，并抛出 `CancelledError`。

        3. **`asyncio.sleep` 成为报错的“触发点”**：
           - 因为后台任务大部分时间都在 `await asyncio.sleep(60)` 休眠，此时取消任务，就会直接中断 `sleep`，抛出 `CancelledError`；
           - 如果取消时，任务恰好正在执行「清理逻辑」（而非休眠），报错会出现在清理逻辑的某一步，但因为清理逻辑执行时间极短（毫秒级），所以几乎所有取消都会命中 `sleep` 阶段，表现为 `sleep` 触发报错。

        简单说：`asyncio.sleep` 是后台任务的「主要等待环节」，也是异步取消机制的「主要触发点」——它不是报错的根源，只是刚好处于“被取消”的位置。


        ### 再补充：为什么取消异步任务会抛这个错？
        这是 Python `asyncio` 的「设计规则」：**异步任务被取消时，必须通过抛出 `CancelledError` 来通知任务“你被取消了”**，让任务有机会执行收尾操作（比如释放资源）。

        - 同步任务被取消（比如线程 `thread.stop()`）是“暴力终止”，不会抛异常；
        - 异步任务的取消是“协作式”的：通过 `task.cancel()` 发送取消信号，任务在执行可取消操作时（如 `sleep`、`await` 其他任务），会响应这个信号，抛出 `CancelledError`，让开发者在 `except` 中处理收尾。

        你的代码中，`clean_expired_pairs` 没有处理 `CancelledError`，所以这个异常会向上传播，最终被 FastAPI 捕获，打印出你看到的报错日志。


        ### 验证：如果没有 `asyncio.sleep`，还会报错吗？
        如果把 `clean_expired_pairs` 改成“无休眠的忙循环”（不推荐，会占满CPU）：
        ```python
        async def clean_expired_pairs(self):
            while True:
                执行清理逻辑...
                # 去掉 asyncio.sleep(60)
        ```
        此时取消任务时，因为没有「可取消的异步操作」，`CancelledError` 会在下次执行 `await` 操作时抛出——但如果是纯忙循环（无任何 `await`），任务会“无法响应取消信号”，一直运行直到服务强制退出，反而会导致资源泄漏。

        所以 `asyncio.sleep` 是必要的（控制清理频率），报错的根源是「任务被取消时未处理异常」，而非 `sleep` 本身。


        ### 最终结论
        - 报错的直接触发点是 `await asyncio.sleep(60)`，但它不是“罪魁祸首”；
        - 核心原因是「服务重载→后台任务被取消→`asyncio.sleep` 响应取消信号抛出 `CancelledError`→未被捕获」；
        - 解决方法很简单：在 `clean_expired_pairs` 中捕获取消异常，让任务优雅退出：
          ```python
          async def clean_expired_pairs(self):
              try:
                  while True:
                      # 原有清理逻辑...
                      await asyncio.sleep(60)
              except asyncio.CancelledError:
                  # 处理收尾（可选），然后静默退出
                  logger.info("后台清理任务已被取消，准备退出")
                  return
          ```

        这样修改后，即使服务重载触发任务取消，也不会打印报错日志，服务会优雅重启。
        """
        """定期清理过期实例（解除绑定，回归空闲）"""
        try:
            while True:
                await asyncio.sleep(30)  # 每半分钟检查一次
                async with self.lock:
                    for agent_pair in self.pool:
                        if agent_pair.get_state() == AgentInstanceState.EXPIRED:
                            # 解除过期会话绑定
                            expired_session = agent_pair.session_id
                            if expired_session in self.session_map:
                                del self.session_map[expired_session]
                            await agent_pair.unbind_session()
                            logger.info(f"清理过期实例绑定，会话 {expired_session} 已释放")

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
        # 设置会话进行中的标识
        SESSION_GOING_CACHE.setdefault(session_id, 1)
        # 2. 构造用户消息
        # user_msg = Msg(name="user", content=question, role="user")
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


# @app.on_event("startup")  # 该方式已过期，采用lifespan代替
# async def startup_event():
#     """服务启动时初始化实例池和后台任务"""
#     await agent_pool.init_pool()
#     asyncio.create_task(agent_pool.clean_expired_pairs())  # 启动过期清理任务
#     logger.info("服务启动完成，等待请求...")


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
    # 校验当前会话是否正在进行中
    is_going = SESSION_GOING_CACHE.get(session_id, 0)
    if is_going == 1:
        return QueryResponse(
            session_id=session_id,
            reply="当前对话尚未结束，请待当前对话完成后进行！",
            timestamp=datetime.now()
        )
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

    uvicorn.run(app='api_for_agent_scope_with_history_message_for_true_scene_multi_person_and_talk_demo:app', port=8090, reload=False,
                workers=1)
