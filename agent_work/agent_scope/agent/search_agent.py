import os
from typing import List
from agentscope.agent import ReActAgent
from agentscope.message import TextBlock
from agentscope.model import ChatModelBase
from agentscope.formatter import DashScopeMultiAgentFormatter
from sentence_transformers import SentenceTransformer
from agentscope.tool import Toolkit, ToolResponse
import faiss
import pickle
import logging
from cachetools import TTLCache  # 带过期时间的缓存，避免内存泄漏
from functools import wraps
from typing import Dict, Any, Callable, Optional
import inspect

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# 缓存配置：key=dialog_msg_id（对话消息ID），value=工具调用次数，过期时间=30分钟（覆盖单次对话最大耗时）
TOOL_CALL_CACHE = TTLCache(maxsize=1000, ttl=1800)  # 最多缓存1000个对话，30分钟无操作自动过期
MAX_TOOL_CALLS_PER_DIALOG = 3


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
            # 从智能体实例缓存中获取当前正在处理的对话id【对应一次对话交互】
            user_msg_list = [msg for msg in agent.memory.content if msg.role == "user" and msg.invocation_id]  # TODO 在添加重写助手后，检索助手接收到的是重写后的问题，并非user信息。带改造，兼容重写助手
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


def create_retriever_agent(chat_model: ChatModelBase) -> ReActAgent:
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
        model=chat_model,
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
