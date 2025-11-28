import traceback
from typing import Any, AsyncGenerator


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