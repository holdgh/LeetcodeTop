from typing import Type

from agentscope.agent import ReActAgent
from agentscope.message import Msg
from pydantic import BaseModel


class BaseAgentForMock(ReActAgent):
    async def reply(
        self,
        msg: Msg | list[Msg] | None = None,
        structured_model: Type[BaseModel] | None = None,
    ) -> Msg:
        # TODO 免费llm到期，采用mock数据
        return Msg(name=self.name, content=f"{self.name}：回复成功！", role="assistant")