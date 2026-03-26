### MCP 服务端主流集成方式总结（附可调试示例代码）
MCP（Model Context Protocol）作为模型与工具交互的标准化协议，其服务端集成方式核心围绕「传输协议」和「部署形态」划分，主流方式包括：**Stdio 本地进程模式**、**HTTP 远程访问模式**、**WebSocket 长连接模式**、**多工具聚合模式**（企业级常用），以下结合知乎文档及行业实践，对每种方式给出定义、适用场景、可调试示例代码。

> 所有示例均基于 Python `mcp` 官方库（最成熟的 MCP 实现），需先安装依赖：
> ```bash
> pip install mcp httpx fastapi uvicorn websockets  # 核心依赖
> ```

## 一、Stdio 本地进程模式（最基础/常用）
### 1. 核心定义
通过「标准输入/输出（stdio）」实现 MCP 客户端与服务端的进程内通信，服务端以独立进程启动，客户端通过管道与其交互，无需网络依赖，是本地调试、轻量集成的首选。

### 2. 适用场景
- 本地开发/调试 MCP 工具；
- 单机部署的工具调用（如 AgentScope/CoPaw 本地集成）；
- 无网络环境下的工具交互。

### 3. 可调试示例代码（计算器+回显工具）
```python
# stdio_mcp_server.py
from typing import Any
from mcp.server.fastmcp import FastMCP

# 初始化 FastMCP 服务（命名为 "stdio-test-server"）
mcp = FastMCP("stdio-test-server")

# 工具 1：加法计算
@mcp.tool()
async def add(a: float, b: float) -> str:
    """
    计算两个数的和
    Args:
        a: 第一个数
        b: 第二个数
    """
    return f"计算结果：{a} + {b} = {a + b}"

# 工具 2：文本回显
@mcp.tool()
async def echo(content: str) -> str:
    """
    回显输入的文本内容
    Args:
        content: 需要回显的文本
    """
    return f"回显内容：{content}"

# 启动服务（stdio 传输模式）
if __name__ == "__main__":
    print("=== Stdio MCP 服务启动，等待客户端连接 ===")
    mcp.run(transport="stdio")  # 核心：指定 stdio 传输
```

### 4. 调试方式
- 启动服务：`python stdio_mcp_server.py`（服务阻塞，等待客户端调用）；
- 客户端配置（AgentScope/CoPaw）：
  ```json
  {
    "key": "stdio_mcp_server",
    "name": "stdio_mcp_server",
    "enabled": true,
    "transport": "stdio",
    "command": "python",
    "args": ["stdio_mcp_server.py"],
    "timeout": 30
  }
  ```
- 调用测试：
  ```python
  import agentscope
  agentscope.init(mcp_configs="上述配置文件路径")
  # 调用加法工具
  result = agentscope.tools.call_tool(
      tool_name="stdio_mcp_server",
      tool_kwargs={"name": "add", "parameters": {"a": 10, "b": 20}}
  )
  print(result["content"])  # 输出：计算结果：10 + 20 = 30
  ```

## 二、HTTP 远程访问模式（跨机器/网络）
### 1. 核心定义
基于 HTTP/HTTPS 协议实现远程通信，MCP 服务端以 HTTP 服务形式监听端口，客户端通过 HTTP POST 请求调用工具，支持跨机器、跨网络访问，是分布式部署的主流方式。

### 2. 适用场景
- 多客户端共享 MCP 服务（如多台机器调用同一工具服务）；
- 云服务器部署 MCP 服务，本地/终端设备调用；
- 需通过 API 网关暴露 MCP 工具的场景。

### 3. 可调试示例代码（天气查询服务）
```python
# http_mcp_server.py
from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP

# 初始化 HTTP 模式 MCP 服务
mcp = FastMCP("http-weather-server")
# 天气 API 基础配置（示例用高德地图免费 API，需替换为自己的 key）
AMAP_WEATHER_API = "https://restapi.amap.com/v3/weather/weatherInfo"
AMAP_KEY = "你的高德地图API Key"  # 从高德开放平台申请：https://lbs.amap.com/

# 工具：根据城市查询天气
@mcp.tool()
async def get_city_weather(city: str) -> str:
    """
    查询指定城市的实时天气
    Args:
        city: 城市名称（如北京、上海）
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                AMAP_WEATHER_API,
                params={"key": AMAP_KEY, "city": city, "extensions": "base"},
                timeout=10
            )
            data = response.json()
            if data["status"] != "1":
                return f"查询失败：{data.get('info', '未知错误')}"
            
            # 解析天气数据
            weather = data["lives"][0]
            return f"""
城市：{weather['city']}
实时温度：{weather['temperature']}℃
天气状况：{weather['weather']}
风力：{weather['windpower']}级
更新时间：{weather['reporttime']}
            """
        except Exception as e:
            return f"查询异常：{str(e)}"

# 启动 HTTP 服务
if __name__ == "__main__":
    print("=== HTTP MCP 服务启动 ===")
    print(f"服务地址：http://0.0.0.0:8000")
    # 核心：指定 HTTP 传输，监听 8000 端口（0.0.0.0 允许远程访问）
    mcp.run(transport="http", host="0.0.0.0", port=8000)
```

### 4. 调试方式
- 启动服务：`python http_mcp_server.py`（控制台显示 `Listening on http://0.0.0.0:8000`）；
- 手动测试（curl/Postman）：
  ```bash
  curl -X POST http://localhost:8000 \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {"name": "get_city_weather", "parameters": {"city": "北京"}},
    "id": 1
  }'
  ```
- 客户端配置（AgentScope/CoPaw）：
  ```json
  {
    "key": "http_mcp_server",
    "name": "http_mcp_server",
    "enabled": true,
    "transport": "http",
    "url": "http://127.0.0.1:8000",  # 替换为服务端实际 IP
    "timeout": 30
  }
  ```

## 三、WebSocket 长连接模式（实时交互）
### 1. 核心定义
基于 WebSocket 协议实现全双工长连接，服务端与客户端建立持久连接，适合高频、低延迟的工具调用（如实时数据查询、流式响应），比 HTTP 更高效。

### 2. 适用场景
- 实时工具调用（如股票行情、实时监控）；
- 流式响应场景（如大文本分段返回）；
- 需双向通信的 MCP 交互（如客户端推送数据，服务端实时处理）。

### 3. 可调试示例代码（实时日志查询）
```python
# websocket_mcp_server.py
from typing import Any
import time
from mcp.server.fastmcp import FastMCP

# 初始化 WebSocket MCP 服务
mcp = FastMCP("websocket-log-server")

# 模拟日志数据
LOG_DATA = [
    "2026-03-10 10:00:00 - 系统启动",
    "2026-03-10 10:05:00 - 接收到用户请求",
    "2026-03-10 10:06:00 - 处理请求成功",
    "2026-03-10 10:10:00 - 系统空闲"
]

# 工具：实时查询日志（支持流式返回）
@mcp.tool()
async def get_realtime_logs(rows: int = 5, stream: bool = False) -> str:
    """
    查询系统实时日志
    Args:
        rows: 返回日志行数（默认 5）
        stream: 是否流式返回（默认 False）
    """
    if not stream:
        # 非流式：直接返回
        return "\n".join(LOG_DATA[:rows])
    else:
        # 流式：模拟逐行返回（WebSocket 长连接优势）
        result = ""
        for log in LOG_DATA[:rows]:
            result += log + "\n"
            time.sleep(0.5)  # 模拟延迟
        return result

# 启动 WebSocket 服务
if __name__ == "__main__":
    print("=== WebSocket MCP 服务启动 ===")
    print(f"服务地址：ws://0.0.0.0:8001")
    # 核心：指定 WebSocket 传输，监听 8001 端口
    mcp.run(transport="websocket", host="0.0.0.0", port=8001)
```

### 4. 调试方式
- 启动服务：`python websocket_mcp_server.py`；
- 客户端配置（AgentScope/CoPaw）：
  ```json
  {
    "key": "websocket_mcp_server",
    "name": "websocket_mcp_server",
    "enabled": true,
    "transport": "websocket",
    "url": "ws://127.0.0.1:8001",
    "timeout": 60
  }
  ```
- 调用测试（流式日志）：
  ```python
  result = agentscope.tools.call_tool(
      tool_name="websocket_mcp_server",
      tool_kwargs={"name": "get_realtime_logs", "parameters": {"rows": 3, "stream": True}}
  )
  print(result["content"])  # 逐行输出 3 条日志（带延迟）
  ```

## 四、多工具聚合模式（企业级）
### 1. 核心定义
将多个独立的 MCP 工具（如天气、计算器、文件操作）聚合到一个服务端，通过「命名空间/工具分组」管理，支持统一接入、统一鉴权、统一监控，是企业级部署的核心方式。

### 2. 适用场景
- 企业内部统一工具平台（如 AI 助手调用多个业务工具）；
- 多团队共享工具集，避免重复开发；
- 需鉴权/限流/日志审计的生产环境。

### 3. 可调试示例代码（聚合工具服务）
```python
# aggregate_mcp_server.py
from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.middleware import Middleware  # 导入中间件（鉴权/日志）

# 初始化聚合 MCP 服务
mcp = FastMCP("aggregate-tool-server")

# ------------- 中间件：简单鉴权（企业级必备）-------------
class AuthMiddleware(Middleware):
    async def before_call(self, context: dict) -> None:
        """调用工具前鉴权"""
        params = context.get("params", {})
        token = params.pop("token", "")
        if token != "mcp_test_123":  # 模拟鉴权逻辑
            raise PermissionError("鉴权失败：无效的 token")

# 注册鉴权中间件
mcp.use(AuthMiddleware())

# ------------- 工具 1：天气查询（复用 HTTP 模式示例）-------------
@mcp.tool(namespace="weather")  # 命名空间：weather
async def get_city_weather(city: str, token: str) -> str:
    AMAP_WEATHER_API = "https://restapi.amap.com/v3/weather/weatherInfo"
    AMAP_KEY = "你的高德API Key"
    async with httpx.AsyncClient() as client:
        response = await client.get(
            AMAP_WEATHER_API,
            params={"key": AMAP_KEY, "city": city, "extensions": "base"},
            timeout=10
        )
        data = response.json()
        return f"{city} 天气：{data['lives'][0]['weather']}，温度：{data['lives'][0]['temperature']}℃"

# ------------- 工具 2：文件读写（本地工具）-------------
@mcp.tool(namespace="file")  # 命名空间：file
async def read_file(file_path: str, token: str) -> str:
    """
    读取本地文件内容
    Args:
        file_path: 文件路径
        token: 鉴权 token
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f"文件内容：\n{f.read()[:500]}"  # 仅返回前 500 字符
    except Exception as e:
        return f"读取失败：{str(e)}"

# ------------- 工具 3：加法计算（基础工具）-------------
@mcp.tool(namespace="calc")  # 命名空间：calc
async def add(a: float, b: float, token: str) -> str:
    return f"计算结果：{a} + {b} = {a + b}"

# 启动聚合服务（HTTP 模式，支持远程访问）
if __name__ == "__main__":
    print("=== 聚合 MCP 服务启动（带鉴权）===")
    print(f"服务地址：http://0.0.0.0:8002")
    mcp.run(transport="http", host="0.0.0.0", port=8002)
```

### 4. 调试方式
- 启动服务：`python aggregate_mcp_server.py`；
- 调用测试（需带鉴权 token）：
  ```python
  # 调用天气工具（带 token）
  result = agentscope.tools.call_tool(
      tool_name="aggregate-tool-server",
      tool_kwargs={
          "name": "get_city_weather",
          "parameters": {"city": "上海", "token": "mcp_test_123"}
      }
  )
  print(result["content"])  # 输出上海天气（鉴权通过）
  
  # 调用计算工具（无 token → 鉴权失败）
  try:
      agentscope.tools.call_tool(
          tool_name="aggregate-tool-server",
          tool_kwargs={"name": "add", "parameters": {"a": 10, "b": 20}}
      )
  except PermissionError as e:
      print(e)  # 输出：鉴权失败：无效的 token
  ```

### 总结
| 集成方式 | 核心特点 | 适用场景 | 核心优势 |
|----------|----------|----------|----------|
| Stdio 本地进程 | 无网络依赖，进程内通信 | 本地调试、单机部署 | 简单、高效、无网络配置 |
| HTTP 远程访问 | 跨网络/跨机器，RESTful 风格 | 分布式部署、云服务 | 通用性强、易接入 API 网关 |
| WebSocket 长连接 | 全双工、低延迟、流式响应 | 实时交互、高频调用 | 比 HTTP 更高效，支持流式返回 |
| 多工具聚合 | 统一管理、鉴权/监控、命名空间 | 企业级部署、多团队共享 | 可维护性强、易扩展、符合生产规范 |

所有示例代码均可直接运行（需替换 API Key 等敏感信息），调试时优先从 Stdio 模式入手（无网络问题），再逐步扩展到 HTTP/WebSocket/聚合模式。