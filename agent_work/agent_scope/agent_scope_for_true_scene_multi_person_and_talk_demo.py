"""
某大型工业设备公司（如机床、发电机组厂商）需构建运维问答 Agent，支持：
    基于运维手册知识库回答设备故障排查、操作规范、保养周期等问题；
    多智能体协作（检索 Agent 负责知识库查询，专家 Agent 负责逻辑推理）；
    工具调用（文档检索、故障代码解析工具）；
    上下文记忆（多轮对话中记住设备型号、历史故障等信息）。
"""
import random
import uuid
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
            toolkit.register_tool_function(doc_retriever.search)
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
import json
from typing import List, Dict, Optional, Any
from agentscope.agent import ReActAgent
from agentscope.message import Msg, TextBlock
from agentscope.pipeline import MsgHub
from agentscope.model import DashScopeChatModel
from agentscope.formatter import DashScopeMultiAgentFormatter
from sentence_transformers import SentenceTransformer
from agentscope.tool import Toolkit, ToolResponse
import faiss
import pickle
import logging

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        with open("maintenance_texts.pkl", "wb") as f:
            pickle.dump(knowledge, f)
        return knowledge

    def _load_texts(self, path: str) -> List[str]:
        """加载知识库文本"""
        with open(path, "rb") as f:
            return pickle.load(f)

    def search(self, query: str, top_k: int = 3) -> ToolResponse:
        # def search(self, query: str, top_k: int = 3) -> List[str]:
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


# 工具2：故障代码解析工具（结构化输出）
# @tool
def parse_error_code(device_model: str, error_code: str) -> ToolResponse:
    # def parse_error_code(device_model: str, error_code: str) -> Dict[str, str]:  # 报错：The tool function must return a ToolResponse object, or an AsyncGenerator/Generator of ToolResponse objects, but got <class 'dict'>.
    """
    解析设备故障代码的详细信息
    Args:
        device_model: 设备型号（如M-2000、M-3000）
        error_code: 报警代码（如E101、E203）
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
                           text=f"未知故障代码，排查步骤：无该{device_model}型号的{error_code}代码记录，请联系技术支持")
    return ToolResponse(content=[result])


# 工具3：保养周期查询工具
# @tool
def query_maintenance_cycle(part_name: str) -> ToolResponse:
    # def query_maintenance_cycle(part_name: str) -> str:
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
class AgentPair:
    """Agent实例对：包含一个检索Agent和一个专家Agent，绑定同一会话"""

    def __init__(self, pair_id: str):
        self.pair_id = pair_id  # 实例对ID
        self.session_id: Optional[str] = None  # 绑定的会话ID（空闲时为None）
        self.bind_time: Optional[datetime] = None  # 绑定会话的时间
        self.idle_timeout = timedelta(minutes=10)  # 空闲超时时间（10分钟无操作则释放）
        self.model = DashScopeChatModel(
            model_name="qwen-max",
            api_key="sk-f61034a0afd64ffdab4be83a063b20e3",
            # api_key=os.getenv("DASHSCOPE_API_KEY"),
            # temperature=0.1  # 降低随机性，保证运维回答准确性
            generate_kwargs={
                "temperature": 0.1,
                "top_p": 0.8,
                "max_tokens": 1500,
                "repetition_penalty": 1.1
            }
        )

        # 创建检索Agent和专家Agent
        self.retriever = self._create_retriever_agent()
        self.expert = self._create_expert_agent()

    def _get_event_loop(self):
        """获取或创建事件循环"""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.new_event_loop()

    def _create_retriever_agent(self) -> ReActAgent:
        # 初始化工具箱实例
        toolkit = Toolkit()
        # 使用 register_tool_function 方法注册工具
        toolkit.register_tool_function(doc_retriever.search)
        toolkit.register_tool_function(parse_error_code)
        toolkit.register_tool_function(query_maintenance_cycle)

        retriever = ReActAgent(
            name="检索助手",
            sys_prompt="""仅处理当前绑定会话的用户问题：
1. 基于专属内存中的对话历史调用工具，结果整理后传递给专家Agent；
2. 不直接回复用户，仅输出工具调用结果；
3. 会话切换时内存会清空，无需考虑历史会话信息。""",
            model=self.model,
            # memory=self.retriever_memory,
            formatter=DashScopeMultiAgentFormatter(),
            toolkit=toolkit
        )
        retriever.set_console_output_enabled(False)  # 禁用控制台输出智能体内容，由业务代码控制输出内容
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
                new_pair = AgentPair(pair_id=f"agent_pair_{len(self.pool) + 1}")
                await new_pair.bind_session(session_id)
                self.pool.append(new_pair)
                self.session_map[session_id] = new_pair
                logger.info(f"实例池扩容，当前容量：{len(self.pool)}")
                return new_pair

            # 4. 已达最大容量→等待空闲实例（超时1分钟）
            logger.warning("实例池已达最大容量，用户会话排队中...")
            return await self._wait_for_idle_pair(session_id)

    async def _wait_for_idle_pair(self, session_id: str, timeout: int = 60) -> AgentPair:
        """等待空闲实例（超时抛出异常）"""
        start_time = datetime.now()
        while datetime.now() - start_time < timedelta(seconds=timeout):
            await asyncio.sleep(2)  # 每2秒检查一次
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
        """定期清理过期实例（解除绑定，回归空闲）"""
        while True:
            await asyncio.sleep(60)  # 每分钟检查一次
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


# -------------------------- 5. 会话管理与多用户服务 --------------------------
class SessionManager:
    @staticmethod
    def create_session() -> str:
        """创建新会话ID"""
        return f"session_{uuid.uuid4().hex[:8]}"


async def user_dialog(session_id: str, agent_pool: AgentPool, questions: List[str]):
    """单个用户多轮对话（复用Agent实例）"""
    logger.info(f"\n【会话 {session_id}】用户开始咨询")
    try:
        for q in questions:
            # 1. 获取绑定该会话的Agent实例对
            agent_pair = await agent_pool.get_agent_pair(session_id)
            retriever = agent_pair.retriever
            expert = agent_pair.expert

            # 2. 构造用户消息
            user_msg = Msg(name="user", content=q, role="user")

            # 3. 存入实例专属内存
            await retriever.memory.add(user_msg)
            await expert.memory.add(user_msg)
            logger.info(f"【会话 {session_id}】发送：{q}")

            # 4. 智能体协作处理
            retriever_reply = await retriever.reply()
            await expert.memory.add(retriever_reply)
            expert_reply = await expert.reply()

            # 5. 输出结果
            logger.info(f"【会话 {session_id}】收到回复：{expert_reply.content}\n")

            # TODO 模拟用户思考间隔（1-3秒）
            await asyncio.sleep(random.random() * 2 + 1)
    except Exception as e:
        logger.error(f"【会话 {session_id}】处理异常：{str(e)}")


async def simulate_non_simultaneous_dialogs(agent_pool: AgentPool):
    """模拟多用户非同时对话（分3批，每批间隔一段时间）"""
    # 第一批用户（同时咨询）
    session1 = SessionManager.create_session()
    session2 = SessionManager.create_session()
    logger.info("=== 第一批用户开始咨询 ===")
    batch1_tasks = [
        asyncio.create_task(user_dialog(session1, agent_pool, [
            "M-2000报警E101，怎么办？",
            "冷却液正常，下一步？",
            "主轴轴承多久保养一次？"
        ])),
        asyncio.create_task(user_dialog(session2, agent_pool, [
            "M-3000报警E056是什么问题？",
            "密封圈怎么检查？"
        ]))
    ]
    await asyncio.gather(*batch1_tasks)
    logger.info("=== 第一批用户咨询结束，等待5秒 ===")
    await asyncio.sleep(5)

    # 第二批用户（第一批实例释放后复用）
    session3 = SessionManager.create_session()
    session4 = SessionManager.create_session()
    logger.info("=== 第二批用户开始咨询 ===")
    batch2_tasks = [
        asyncio.create_task(user_dialog(session3, agent_pool, [
            "设备开机前要检查什么？",
            "冷却滤芯更换周期？",
            "液压油多久换一次？"
        ])),
        asyncio.create_task(user_dialog(session4, agent_pool, [
            "M-2000的最大加工转速是多少？",
            "变频器多久校准一次？"
        ]))
    ]
    await asyncio.gather(*batch2_tasks)
    logger.info("=== 第二批用户咨询结束，等待5秒 ===")
    await asyncio.sleep(5)

    # 第三批用户（高并发场景，触发实例池扩容）
    session5 = SessionManager.create_session()
    session6 = SessionManager.create_session()
    session7 = SessionManager.create_session()
    session8 = SessionManager.create_session()
    logger.info("=== 第三批用户开始咨询（高并发） ===")
    batch3_tasks = [
        asyncio.create_task(user_dialog(session5, agent_pool, [
            "M-2000报警E203怎么处理？",
            "电机接线怎么检查？"
        ])),
        asyncio.create_task(user_dialog(session6, agent_pool, [
            "安全防护门多久检查一次？",
            "检查时需要注意什么？"
        ])),
        asyncio.create_task(user_dialog(session7, agent_pool, [
            "M-3000的额定负载是多少？",
            "超过负载会有什么后果？"
        ])),
        asyncio.create_task(user_dialog(session8, agent_pool, [
            "主轴轴承润滑脂更换步骤？",
            "用什么型号的润滑脂？"
        ]))
    ]
    await asyncio.gather(*batch3_tasks)
    logger.info("=== 第三批用户咨询结束 ===")

# -------------------------- 主函数：启动服务 --------------------------
async def main():
    # 1. 初始化Agent实例池（预创建5个实例对，最大扩容到20个）
    agent_pool = AgentPool(min_size=5, max_size=20)
    await agent_pool.init_pool()

    # 2. 启动后台任务：定期清理过期实例绑定、缩容
    asyncio.create_task(agent_pool.clean_expired_pairs())

    # 3. 模拟多用户非同时对话
    await simulate_non_simultaneous_dialogs(agent_pool)

    # 4. 等待后台任务结束（实际服务中可无限运行）
    logger.info("=== 所有用户咨询结束，服务继续运行（按Ctrl+C终止） ===")
    try:
        while True:
            await asyncio.sleep(3600)  # 持续运行
    except KeyboardInterrupt:
        logger.info("=== 服务正在关闭 ===")

if __name__ == "__main__":
    asyncio.run(main())
