#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2026/3/26 14:04
# @Author  : gaohuan
# @Email   : 
# @FileName: one.py
# @Desc    :
import docker
import time
import math
from fastapi import FastAPI, Request, Response
from fastapi.responses import RedirectResponse
import uvicorn
import threading
from collections import defaultdict

# ====================== 配置 ======================
APP_IMAGE = "osclaw:v010.1"  # 例如 osclaw:v010.1
APP_PORT = 8088  # 容器内部端口
MAX_CONCURRENCY_PER_INSTANCE = 1  # 单个实例最大并发
CHECK_INTERVAL = 5  # 每5秒检查一次是否扩容
LB_PORT = 8000  # 对外统一入口

# ====================== 状态 ======================
current_requests = 0
instance_list = []  # 运行中的容器列表
client = docker.from_env()


# ====================== Docker 操作 ======================
def start_new_instance():
    """启动一个新的应用A实例"""
    container = client.containers.run(
        image=APP_IMAGE,
        ports={f"{APP_PORT}/tcp": None},  # 随机主机端口
        detach=True,
        restart_policy={"Name": "on-failure"},
        mem_limit='1g',  # 内存限制
        cpu_quota=50000  # CPU限制 (50%)
    )
    container.reload()
    host_port = container.attrs['NetworkSettings']['Ports'][f'{APP_PORT}/tcp'][0]['HostPort']
    instance_list.append({
        "container": container,
        "port": host_port,
        "current_load": 0
    })
    print(f"✅ 启动新实例：localhost:{host_port}")


def stop_instance(idx):
    """停止一个实例"""
    info = instance_list.pop(idx)
    info["container"].stop()
    info["container"].remove()
    print(f"❌ 销毁实例：localhost:{info['port']}")


# ====================== 自动扩缩容核心 ======================
def auto_scaler_loop():
    global current_requests
    while True:
        if not current_requests:
            time.sleep(CHECK_INTERVAL)
            continue

        # 计算需要多少实例
        required = math.ceil(current_requests / MAX_CONCURRENCY_PER_INSTANCE)
        current = len(instance_list)

        print(f"📊 请求数:{current_requests} 单实例上限:{MAX_CONCURRENCY_PER_INSTANCE}")
        print(f"📦 需要实例:{required} 当前实例:{current}")

        # 扩容
        while required > current:
            start_new_instance()
            current += 1

        # 缩容（负载低时销毁多余实例）
        while required < current and current > 1:
            stop_instance(0)
            current -= 1

        time.sleep(CHECK_INTERVAL)


# ====================== 负载均衡 ======================
def get_next_instance():
    """最简单的轮询负载均衡"""
    if not instance_list:
        start_new_instance()
    instance = instance_list[0]
    instance_list.append(instance_list.pop(0))
    return instance


# ====================== 统一入口 ======================
app = FastAPI()


@app.middleware("http")
async def count_requests(request: Request, call_next):
    """全局请求计数（用于扩容计算）"""
    global current_requests
    current_requests += 1
    try:
        return await call_next(request)
    finally:
        current_requests -= 1


@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def forward(request: Request, full_path: str):
    """转发所有请求到后端容器"""
    inst = get_next_instance()
    url = f"http://127.0.0.1:{inst['port']}/{full_path}"

    # 简单转发（你也可以用 httpx 完整转发）
    return RedirectResponse(url=url)


# ====================== 启动 ======================
if __name__ == "__main__":
    print("🚀 启动自动扩缩容负载均衡器")
    threading.Thread(target=auto_scaler_loop, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=LB_PORT)
