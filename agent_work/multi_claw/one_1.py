import docker
import time
import math
import threading
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import httpx

# ====================== 配置 ======================
APP_IMAGE = "osclaw:v010.1"
APP_PORT = 8088
MAX_CONCURRENCY_PER_INSTANCE = 1
CHECK_INTERVAL = 3
LB_PORT = 8000
TIMEOUT = 30

# ====================== 状态 ======================
active_requests = 0
instance_pool = []
docker_client = docker.from_env()


# ====================== Docker 实例管理 ======================
def start_instance():
    container = docker_client.containers.run(
        image=APP_IMAGE,
        ports={f"{APP_PORT}/tcp": None},
        detach=True
    )
    container.reload()
    host_port = container.attrs["NetworkSettings"]["Ports"][f"{APP_PORT}/tcp"][0]["HostPort"]
    instance = {
        "container": container,
        "host_port": host_port,
        "load": 0
    }
    instance_pool.append(instance)
    print(f"✅ 启动新实例 -> :{host_port}")


def stop_instance(index):
    inst = instance_pool.pop(index)
    inst["container"].stop()
    inst["container"].remove()
    print(f"❌ 销毁实例 -> :{inst['host_port']}")


# ====================== 自动扩缩容 ======================
def auto_scale_loop():
    global active_requests
    while True:
        if not instance_pool:
            start_instance()
            time.sleep(1)
            continue

        req = active_requests
        required = math.ceil(req / MAX_CONCURRENCY_PER_INSTANCE)
        required = max(required, 1)
        current = len(instance_pool)

        print(f"📊 活跃请求: {req} | 当前实例: {current} | 需要实例: {required}")

        while required > current:
            start_instance()
            current += 1
        while required < current and current > 1:
            stop_instance(0)
            current -= 1

        time.sleep(CHECK_INTERVAL)


# ====================== 轮询负载均衡 ======================
round_robin_index = 0


def get_next_instance():
    global round_robin_index
    if not instance_pool:
        start_instance()
    inst = instance_pool[round_robin_index % len(instance_pool)]
    round_robin_index += 1
    return inst


# ====================== ✅ 核心修复：完整路径代理 ======================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT))


@app.middleware("http")
async def counter_middleware(request: Request, call_next):
    global active_requests
    active_requests += 1
    try:
        return await call_next(request)
    finally:
        active_requests -= 1


# ✅ 关键修复：使用 request.url.path 完整保留原始路径
@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"])
async def proxy(request: Request, full_path: str):
    inst = get_next_instance()

    # ✅ 这一行是修复 404 的核心：完整保留路径 + 查询参数
    target_url = f"http://127.0.0.1:{inst['host_port']}{request.url.path}"
    if request.url.query:
        target_url += f"?{request.url.query}"

    # 转发所有内容
    req_content = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ["host", "connection"]}

    response = await client.request(
        method=request.method,
        url=target_url,
        headers=headers,
        content=req_content
    )

    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=dict(response.headers),
    )


# ====================== 启动 ======================
if __name__ == "__main__":
    import uvicorn

    threading.Thread(target=auto_scale_loop, daemon=True).start()
    print("🚀 负载均衡 + 自动扩缩容 + 完整代理已启动 (端口 8000)")
    uvicorn.run(app, host="0.0.0.0", port=LB_PORT)