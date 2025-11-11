# 搭建fastapi接口
from fastapi import FastAPI

app = FastAPI()


@app.get("/api/query")
def query_data(key: str):
    return {"result": f"处理请求：{key}"}


if __name__ == '__main__':
    import uvicorn
    """
    当 reload=True 时，Uvicorn 需要监控文件变化并重新加载整个应用。
    如果直接传入应用对象，Uvicorn 在重载过程中无法重新执行你的脚本来再次实例化这个对象。
    而以字符串形式指定应用路径，Uvicorn 就能在每次需要重载时，通过导入该字符串指定的模块来正确获取应用实例。
    """
    # 在使用reload时，app属性值必须为字符串路径名【你的 Python 文件名（不含 .py 后缀）:文件内创建的 FastAPI 应用实例的变量名】
    uvicorn.run(app='fast_api_demo:app', host='0.0.0.0', port=8088, reload=True)
