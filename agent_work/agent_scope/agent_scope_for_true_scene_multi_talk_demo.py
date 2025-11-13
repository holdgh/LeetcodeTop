"""
某大型工业设备公司（如机床、发电机组厂商）需构建运维问答 Agent，支持：
    基于运维手册知识库回答设备故障排查、操作规范、保养周期等问题；
    多智能体协作（检索 Agent 负责知识库查询，专家 Agent 负责逻辑推理）；
    工具调用（文档检索、故障代码解析工具）；
    上下文记忆（多轮对话中记住设备型号、历史故障等信息）。
"""
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
import asyncio
import os
import json
from typing import List, Dict
from agentscope.agent import ReActAgent
from agentscope.message import Msg, TextBlock
from agentscope.pipeline import MsgHub
from agentscope.model import DashScopeChatModel
from agentscope.formatter import DashScopeMultiAgentFormatter
from sentence_transformers import SentenceTransformer
from agentscope.tool import Toolkit, ToolResponse
import faiss
import pickle


# -------------------------- 1. 工具定义（招聘要求：工具调用能力）--------------------------
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
        result = TextBlock(type='text', text=f"未知故障代码，排查步骤：无该{device_model}型号的{error_code}代码记录，请联系技术支持")
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
    return ToolResponse(content=[TextBlock(type='text', text=cycle_db.get(part_name, f"无{part_name}的保养周期记录，请参考运维手册"))])


# 初始化工具实例
doc_retriever = MaintenanceDocRetriever()

# 正确的初始化方式
toolkit = Toolkit()
# 使用 register_tool_function 方法注册工具
toolkit.register_tool_function(doc_retriever.search)
toolkit.register_tool_function(parse_error_code)
toolkit.register_tool_function(query_maintenance_cycle)

# -------------------------- 2. 智能体定义（招聘要求：多Agent协作、Planning、Memory）--------------------------
def create_maintenance_agents() -> List[ReActAgent]:
    """创建运维场景多智能体：检索Agent、专家Agent"""
    # 模型配置（使用通义千问，支持本地模型替换）
    model = DashScopeChatModel(
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

    # 智能体1：检索Agent（负责知识库查询和工具调用）
    retriever_agent = ReActAgent(
        name="检索助手",
        sys_prompt="""你是设备运维检索专家，负责：
1. 接收用户问题后，优先使用运维手册检索工具获取相关知识；
2. 若用户提供设备型号和故障代码，调用故障代码解析工具；
3. 若用户查询保养周期，调用保养周期查询工具；
4. 工具返回结果后，整理成简洁的信息，传递给专家助手做进一步分析；
5. 记牢用户提到的设备型号、故障历史等信息，后续对话复用。""",
        model=model,
        formatter=DashScopeMultiAgentFormatter(),
        # 绑定工具
        # tools=[
        #     Tool(func=lambda query: doc_retriever.search(query), name="运维手册检索",
        #          description="查询设备故障、操作规范、保养周期等知识"),
        #     Tool(func=parse_error_code, name="故障代码解析", description="解析特定设备型号的报警代码"),
        #     Tool(func=query_maintenance_cycle, name="保养周期查询", description="查询设备部件的保养周期")
        # ]
        # 在 AgentScope 1.0.7 版本中，ReActAgent 使用的是 toolkit 实例【在初始化后，需要将工具手动注册到该实例中】
        toolkit=toolkit,  # TODO 如何将工具说明放入工具箱中呢？
        # print_hint_msg=True # 是否打印提示消息，包括推理提示来自计划笔记本，检索信息来自长期记忆和知识库。
    )
    retriever_agent.set_console_output_enabled(False)  # 禁用控制台输出智能体内容，由业务代码控制输出内容

    # 智能体2：专家Agent（负责逻辑推理、故障定位、解决方案生成）
    expert_agent = ReActAgent(
        name="运维专家",
        sys_prompt="""你是大型工业设备运维专家，拥有10年经验，负责：
1. 接收检索助手提供的知识和工具结果，结合运维经验给出精准解决方案；
2. 若信息不足，向用户追问关键信息（如设备型号、故障现象、运行时长）；
3. 回答需包含：故障原因、排查步骤（按优先级排序）、解决方案、注意事项；
4. 多轮对话中记住用户的设备情况，避免重复提问；
5. 语言通俗易懂，步骤清晰，适合现场运维人员操作。""",
        model=model,
        formatter=DashScopeMultiAgentFormatter()
    )
    expert_agent.set_console_output_enabled(False)  # 禁用控制台输出智能体内容，由业务代码控制输出内容

    return [retriever_agent, expert_agent]


# -------------------------- 3. 多Agent协作流程（招聘要求：Workflow编排、上下文管理）--------------------------
# async def main():
#     """运维问答Agent主流程"""
#     # 创建智能体
#     retriever_agent, expert_agent = create_maintenance_agents()
#
#     # 初始化消息枢纽，设置对话主题
#     async with MsgHub(
#             participants=[retriever_agent, expert_agent],
#             announcement=Msg(
#                 name="user",
#                 content="""你好！我是设备运维人员，现在遇到问题：我的设备开机后报警代码E101，设备型号是M-2000，已经运行了1800小时，
# 之前没有出现过类似故障，请问该怎么处理？另外想了解一下主轴轴承的保养周期。""",
#                 role="user",
#             ),
#     ) as hub:
#         # 第一轮：检索Agent查询知识和工具
#         print("=== 第一轮：检索助手工作 ===")
#         await retriever_agent()
#
#         # 第二轮：专家Agent分析并给出解决方案
#         print("\n=== 第二轮：运维专家解答 ===")
#         await expert_agent()
#
#         # 多轮对话：用户追问后续问题
#         print("\n=== 第三轮：用户追问 ===")
#         user_follow_msg = Msg(
#             name="user",
#             content="按照你的方法排查后，冷却液液位是正常的，水泵也没有异响，接下来该怎么办？",
#             role="user",
#         )
#         # 发送追问消息并驱动智能体响应 在 AgentScope 1.0.7 中，应该使用 hub.broadcast() 方法来发送消息
#         await hub.broadcast(user_follow_msg)
#         await retriever_agent()
#         await expert_agent()


async def main():
    """支持用户多次对话的运维问答Agent主流程"""
    # 创建智能体（全局唯一，持续监听）
    retriever_agent, expert_agent = create_maintenance_agents()
    # 初始化消息枢纽，设置欢迎语
    print("=== 工业设备运维问答Agent ===")
    print("欢迎咨询设备故障排查、操作规范、保养周期等问题（输入'quit'退出）")
    print("示例提问：我的M-2000设备报警E101，该怎么处理？\n")
    # 启动MsgHub
    async with MsgHub(participants=[retriever_agent, expert_agent]) as hub:

        # 循环监听用户输入（无限次对话）
        while True:
            # 接收用户输入（支持多行文本，按Enter提交）
            user_input = input("\n你：")
            if user_input.strip().lower() == "quit":
                print("运维专家：感谢咨询，祝你工作顺利！")
                break
            if not user_input.strip():
                print("运维专家：请输入具体问题，我会为你解答~")
                continue

            # 1. 封装用户消息并发送到MsgHub
            user_msg = Msg(
                name="user",
                content=user_input.strip(),
                role="user",
            )
            await hub.broadcast(user_msg)
            print(f"\n=== 智能体处理中... ===")

            # 2. 触发智能体协作（检索→专家解答）
            await retriever_agent()  # 检索Agent工具调用+信息整理
            await expert_agent()  # 专家Agent逻辑推理+生成回答

            # 3. 提取并打印专家最终回答（从MsgHub历史中获取最新响应）
            # 筛选专家Agent发送给用户的最新消息
            expert_responses = await expert_agent.memory.get_memory()
            if expert_responses:
                latest_response = expert_responses[-1]
                print(f"\n运维专家：{latest_response.content}")


# -------------------------- 4. 运行程序 --------------------------
if __name__ == "__main__":
    # 确保环境变量配置了API密钥（Windows：set DASHSCOPE_API_KEY=你的密钥；Linux/Mac：export DASHSCOPE_API_KEY=你的密钥）
    # if not os.getenv("DASHSCOPE_API_KEY"):
    #     print("请先配置环境变量 DASHSCOPE_API_KEY（从阿里云获取）")
    #     exit(1)
    # 安装依赖提示（首次运行需执行）
    required_packages = ["agentscope==1.0.7", "dashscope==1.14.0", "sentence-transformers", "faiss-cpu", "pickle-mixin"]
    asyncio.run(main())
