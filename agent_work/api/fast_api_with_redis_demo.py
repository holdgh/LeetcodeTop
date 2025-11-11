# redis缓存高频请求
from fastapi import FastAPI
import redis

app = FastAPI()
r = redis.Redis(host='localhost', port=6379, db=2)


@app.get("/api/query")
def query_data(key: str):
    # 先查缓存
    cache_res = r.get(key)
    if cache_res:
        return {"result": cache_res.decode(), "from": "cache"}
    # 缓存不存在，计算结果后写入
    res = f"处理请求：{key}"
    r.set(key, res, ex=300)  # 缓存5分钟
    return {"result": res, "from": "calc"}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app='fast_api_with_redis_demo:app', port=8089, reload=True)