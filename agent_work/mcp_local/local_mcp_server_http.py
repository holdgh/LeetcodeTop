#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2026/3/9 15:58
# @Author  : gaohuan
# @Email   : 
# @FileName: local_mcp_server.py
# @Desc    :
#!/usr/bin/env python3
# http_mcp_server_v1.21.0.py
from typing import Any
import httpx
from mcp import Server  # 1.21.0 核心导入（替代原 FastMCP）
from mcp.http import HTTPServer  # HTTP 服务端核心类

# -------------------------- 1. 初始化 MCP 服务 --------------------------
# 创建 Server 实例（替代原 FastMCP）
server = Server("http-weather-server")

# -------------------------- 2. 定义工具函数 --------------------------
# 天气 API 配置（替换为自己的高德 Key）
AMAP_WEATHER_API = "https://restapi.amap.com/v3/weather/weatherInfo"
AMAP_KEY = "你的高德地图API Key"  # 高德开放平台申请：https://lbs.amap.com/


# 注册工具（装饰器改为 @server.method）
@server.method()
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


# -------------------------- 3. 启动 HTTP 服务 --------------------------
if __name__ == "__main__":
    print("=== HTTP MCP 服务启动（mcp==1.21.0）===")
    # 核心：创建 HTTPServer，指定 host/port，绑定 Server 实例
    http_server = HTTPServer(server, host="0.0.0.0", port=8000)
    # 启动服务（阻塞运行）
    http_server.serve()