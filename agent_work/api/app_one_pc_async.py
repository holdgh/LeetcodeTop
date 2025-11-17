import uuid
import asyncio  # 替换time，使用异步睡眠
from fastapi import FastAPI


app = FastAPI()


# 关键点1：定义为异步函数（async def）
@app.post("/api/work")
async def work():
    # 关键点2：用异步IO操作替换同步阻塞（asyncio.sleep替代time.sleep）
    await http_mock()  # 模拟异步IO操作（不会阻塞线程）
    await mysql_query_mock()
    await io_mock()
    return {"message": "下单成功", "order_no": str(uuid.uuid4())}


async def http_mock():
    await asyncio.sleep(3)
    return 1


async def mysql_query_mock():
    await asyncio.sleep(3)
    return 1


async def io_mock():
    await asyncio.sleep(3)
    return 1


if __name__ == "__main__":
    import uvicorn

    # 关键点3：使用异步工作器（uvicorn默认使用异步模式）
    # workers参数：根据CPU核心数设置（通常=核心数*2）
    # reload=True 是 Uvicorn 提供的代码热重载功能，主要作用是：当检测到项目中的代码文件（如 Python 脚本、模板等）发生修改时，自动重启 Uvicorn 服务，无需手动停止再启动。
    # 等待队列参数：backlog=1000（队列最大长度，足够容纳 1000 个等待请求，暂不成为瓶颈）；
    uvicorn.run("app_one_pc_async:app", host="0.0.0.0", port=8081, workers=8, reload=False, backlog=2100)
