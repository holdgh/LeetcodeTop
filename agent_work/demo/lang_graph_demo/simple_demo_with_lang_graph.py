import os
import traceback
from typing import Dict, Any, List, AsyncGenerator

from agentscope.model import DashScopeChatModel
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import MessagesState
import asyncio

from langgraph.graph.state import CompiledStateGraph


# 设置OpenAI API密钥 (请替换为你的实际密钥)
# os.environ["OPENAI_API_KEY"] = "your-openai-api_demo-key"

async def parse_model_response(response: Any) -> str:
    response_text = "抱歉，暂时无法生成结果"
    try:
        if isinstance(response, AsyncGenerator):
            async for content_chunk in response:
                response_text = content_chunk.content
            if response_text and isinstance(response_text, list):
                response_text = response_text[0]['text']
        elif response:
            response_text = response.content
    except Exception as e:
        traceback.print_exc()
        response_text = f"抱歉，模型响应解析异常：{e}"
    return response_text


class SimpleQAAgent:
    """基于LangGraph的简易问答智能体"""

    def __init__(self):
        """
        初始化智能体
        """
        # self.model = ChatOpenAI(model=model_name, temperature=0.7)
        self.model = DashScopeChatModel(
            model_name="deepseek-v3",
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
        self.memory = MemorySaver()  # 使用内存检查点，避免SQLite依赖
        self.graph = self._build_agent_graph()

    def _build_agent_graph(self) -> CompiledStateGraph[Any, Any, Any, Any]:
        """构建智能体的工作流图"""

        # 定义状态结构
        class AgentState(MessagesState):
            """智能体状态，继承自MessagesState以自动处理消息历史"""
            current_query: str = ""  # 当前查询
            response: str = ""  # 生成的响应
            confidence: float = 0.0  # 回答置信度

        # 创建图
        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node("process_query", self._process_query_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("evaluate_confidence", self._evaluate_confidence_node)

        # 设置边和入口点
        workflow.set_entry_point("process_query")

        # 定义工作流
        workflow.add_edge("process_query", "generate_response")
        workflow.add_edge("generate_response", "evaluate_confidence")
        workflow.add_edge("evaluate_confidence", END)

        # 编译图
        return workflow.compile(checkpointer=self.memory)

    def _process_query_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """处理查询节点 - 分析和理解用户问题"""
        print("🔍 正在分析用户问题...")

        current_messages = state["messages"]
        latest_message = current_messages[-1]

        # 提取当前查询
        current_query = latest_message.content if hasattr(latest_message, 'content') else str(latest_message)

        # 简单的查询分类
        query_type = self._classify_query(current_query)

        return {
            "current_query": current_query,
            "query_type": query_type,
            "messages": current_messages  # 保持消息历史不变
        }

    async def _generate_response_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """生成响应节点 - 基于查询生成回答"""
        print("🤖 正在生成回答...")

        current_query = state["current_query"]
        query_type = state.get("query_type", "general")
        messages = state["messages"]

        # 根据查询类型定制系统提示
        system_prompt = self._get_system_prompt(query_type)

        # 构建完整的消息列表
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})

        # 添加历史消息和当前查询
        for msg in messages:
            if hasattr(msg, 'content'):
                role = "user" if hasattr(msg, 'type') and msg.type == 'human' else "assistant"
                full_messages.append({"role": role, "content": msg.content})

        # 调用LLM生成响应
        response_text = None
        try:
            response = await self.model(full_messages)
            response_text = await parse_model_response(response)
            # response_text = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            response_text = f"抱歉，生成回答时出现错误: {str(e)}"

        # 添加AI消息到状态
        new_messages = messages + [AIMessage(content=response_text)]

        return {
            "response": response_text,
            "messages": new_messages
        }

    async def _evaluate_confidence_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """评估置信度节点 - 评估回答的置信度"""
        print("📊 正在评估回答质量...")

        response = state["response"]
        current_query = state["current_query"]

        # 简单的置信度评估
        confidence_prompt = f"""
        请评估以下回答对于问题的匹配程度，给出一个0-1之间的置信度分数：

        问题: {current_query}
        回答: {response}

        只返回一个0-1之间的数字，不要其他文本。
        """
        try:
            confidence_response = await self.model([{"role": "user", "content": confidence_prompt}])
            confidence_text = await parse_model_response(confidence_response)
            # confidence_text = confidence_response.content if hasattr(confidence_response, 'content') else str(
            #     confidence_response)

            # 尝试提取数字
            try:
                confidence = float(confidence_text.strip())
                confidence = max(0.0, min(1.0, confidence))  # 确保在0-1范围内
            except ValueError:
                confidence = 0.5  # 默认值
        except Exception:
            confidence = 0.5

        return {
            "confidence": confidence,
            "messages": state["messages"]  # 保持消息历史
        }

    def _classify_query(self, query: str) -> str:
        """简单的查询分类"""
        query_lower = query.lower()

        if any(word in query_lower for word in ['什么', '是什么', '定义', '解释']):
            return "definition"
        elif any(word in query_lower for word in ['怎么', '如何', '步骤', '方法']):
            return "howto"
        elif any(word in query_lower for word in ['为什么', '原因', '为何']):
            return "why"
        elif any(word in query_lower for word in ['例子', '示例', '举例']):
            return "example"
        else:
            return "general"

    def _get_system_prompt(self, query_type: str) -> str:
        """根据查询类型获取系统提示"""
        prompts = {
            "definition": "你是一个专业的解释助手。请用清晰、准确的语言解释概念，提供相关的背景信息。",
            "howto": "你是一个步骤指导专家。请提供详细、可操作的步骤说明，确保用户能够按照指导完成任务。",
            "why": "你是一个原因分析专家。请深入解释现象背后的原因，提供多角度的分析。",
            "example": "你是一个示例提供专家。请提供具体、相关的例子来帮助理解，确保例子贴近实际应用。",
            "general": "你是一个有帮助的AI助手。请用友好、专业的语气回答问题，确保信息准确有用。"
        }
        return prompts.get(query_type, prompts["general"])

    async def ask_question(self, question: str, thread_id: str = "default") -> Dict[str, Any]:
        """
        向智能体提问

        Args:
            question: 用户问题
            thread_id: 对话线程ID，用于维护对话历史

        Returns:
            包含回答和元数据的字典
        """
        print(f"\n💬 用户提问: {question}")

        # 准备输入状态
        input_state = {
            "messages": [HumanMessage(content=question)]
        }

        try:
            # 执行图工作流
            config = {"configurable": {"thread_id": thread_id}}
            final_state = await self.graph.ainvoke(input_state, config=config)

            # 提取结果
            result = {
                "answer": final_state.get("response", "抱歉，无法生成回答。"),
                "confidence": final_state.get("confidence", 0.0),
                "query_type": final_state.get("query_type", "general"),
                "thread_id": thread_id
            }

            print(f"✅ 回答生成完成 (置信度: {result['confidence']:.2f})")
            return result

        except Exception as e:
            error_msg = f"处理问题时出现错误: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "answer": error_msg,
                "confidence": 0.0,
                "query_type": "error",
                "thread_id": thread_id
            }

    def get_conversation_history(self, thread_id: str = "default") -> List[Dict]:
        """获取对话历史"""
        history = []
        try:
            snapshot = self.memory.get(config={"configurable": {"thread_id": thread_id}})
            if snapshot and 'channel_values' in snapshot and 'messages' in snapshot['channel_values']:
                messages = snapshot['channel_values']['messages']
                for msg in messages:
                    if hasattr(msg, 'content'):
                        role = "用户" if hasattr(msg, 'type') and msg.type == 'human' else "助手"
                        history.append({"role": role, "content": msg.content})
        except Exception:
            traceback.print_exc()
        finally:
            return history


# 使用示例
async def main():
    """主函数 - 演示智能体使用"""

    # 创建智能体实例
    print("🚀 初始化问答智能体...")
    agent = SimpleQAAgent()

    # 测试问题
    test_questions = [
        "人工智能是什么？",
        # "如何学习Python编程？",
        # "为什么天空是蓝色的？",
        # "请给我一个机器学习的例子"
    ]

    # 逐个提问
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'=' * 50}")
        print(f"第 {i} 个问题")

        result = await agent.ask_question(question, thread_id="test_session")

        print(f"\n📝 问题: {question}")
        print(f"🤖 回答: {result['answer']}")
        print(f"📊 置信度: {result['confidence']:.2f}")
        print(f"🏷️  问题类型: {result['query_type']}")

    # 显示对话历史
    print(f"\n{'=' * 50}")
    print("💾 对话历史:")
    history = agent.get_conversation_history("test_session")
    for msg in history:
        print(f"{msg['role']}: {msg['content']}")


if __name__ == "__main__":
    # 运行演示
    asyncio.run(main())