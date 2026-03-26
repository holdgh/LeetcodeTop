#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2026/3/25 9:51
# @Author  : gaohuan
# @Email   : 
# @FileName: main.py
# @Desc    :
# manager/main.py
import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
# 新增静态文件相关导入
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
from docker_multi_claw_for_user.claw_instance_manager import CoPawInstanceManager
from docker_multi_claw_for_user.claw_lifespan_manager import CoPawLifecycleManager
from docker_multi_claw_for_user.data_backup_recover import CoPawBackupManager
from docker_multi_claw_for_user.resource_monitor import CoPawResourceMonitor
from load_balancer import LoadBalancer

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局管理器实例 TODO 暂时采用全局初始化方式
# instance_manager = None
# lifecycle_manager = None
# resource_monitor = None
# backup_manager = None
# load_balancer = None

# 初始化管理器
instance_manager = CoPawInstanceManager()
load_balancer = LoadBalancer()
lifecycle_manager = CoPawLifecycleManager(instance_manager)
resource_monitor = CoPawResourceMonitor(instance_manager)
backup_manager = CoPawBackupManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global instance_manager, lifecycle_manager, resource_monitor, backup_manager, load_balancer

    # 启动时初始化
    logger.info("Starting CoPaw Multi-User Manager...")
    # TODO 暂时注释，放置全局，以便查看代码
    # # 初始化管理器
    # instance_manager = CoPawInstanceManager()
    # load_balancer = LoadBalancer()
    # lifecycle_manager = CoPawLifecycleManager(instance_manager)
    # resource_monitor = CoPawResourceMonitor(instance_manager)
    # backup_manager = CoPawBackupManager()

    # 设置负载均衡器
    lifecycle_manager.set_load_balancer(load_balancer)

    # 启动监控
    lifecycle_manager.start_health_monitoring()
    resource_monitor.start_monitoring()

    logger.info("CoPaw Multi-User Manager started successfully")

    yield

    # 关闭时清理
    logger.info("Shutting down CoPaw Multi-User Manager...")
    resource_monitor.stop_monitoring()
    logger.info("CoPaw Multi-User Manager stopped")


# 创建FastAPI应用
app = FastAPI(
    title="CoPaw Multi-User Manager",
    description="多用户CoPaw实例管理系统",
    version="1.0.0",
    lifespan=lifespan
)

# 获取当前 main.py 所在的文件夹绝对路径
BASE_DIR = Path(__file__).resolve().parent

# 静态文件夹绝对路径
STATIC_DIR = BASE_DIR / "static"
# 挂载静态文件目录
# ✅ 正确：使用绝对路径（Windows/Linux 都兼容）
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# 前端页面路由
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = STATIC_DIR / "index.html"
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API路由
@app.post("/admin/users/{user_id}/register")
async def register_user(user_id: str, user_name: str):
    """用户注册"""
    try:
        result = lifecycle_manager.on_user_register(user_id, user_name)
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/users/{user_id}/login")
async def login_user(user_id: str):
    """用户登录"""
    try:
        url = lifecycle_manager.on_user_login(user_id)
        if url:
            return {"success": True, "url": url}
        else:
            raise HTTPException(status_code=404, detail="Instance not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/users/{user_id}/logout")
async def logout_user(user_id: str):
    """用户登出"""
    try:
        lifecycle_manager.on_user_logout(user_id)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/admin/users/{user_id}")
async def delete_user(user_id: str):
    """删除用户"""
    try:
        lifecycle_manager.on_user_delete(user_id)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/instances")
async def list_instances():
    """列出所有实例"""
    try:
        instances = instance_manager.list_instances()
        return {"success": True, "data": instances}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/backups/{user_id}")
async def list_user_backups(user_id: str):
    """列出用户备份"""
    try:
        backups = backup_manager.list_backups(user_id)
        return {"success": True, "data": backups}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/backups/{user_id}/create")
async def create_backup(user_id: str):
    """创建用户备份"""
    try:
        backup_path = backup_manager.backup_user_data(user_id, instance_manager)
        return {"success": True, "backup_path": backup_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/metrics/{user_id}")
async def get_user_metrics(user_id: str):
    """获取用户资源指标"""
    try:
        metrics = resource_monitor.get_metrics_summary(user_id)
        return {"success": True, "data": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "service": "copaw-manager"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
