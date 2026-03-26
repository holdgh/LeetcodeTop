#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import time
from typing import Dict

import requests


# @Time    : 2026/3/26 11:19
# @Author  : gaohuan
# @Email   : 
# @FileName: port_is_use.py
# @Desc    :
def is_port_available_old(port: int) -> bool:
    """
    检查端口是否可用（Windows/Linux 通用，真正准确）
    """
    import socket

    try:
        # 测试 0.0.0.0（全网卡），这是服务真正使用的绑定方式
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('0.0.0.0', port))
        return True

    except PermissionError:
        # 权限不足（Windows 常见于 <1024 端口）
        return False

    except OSError:
        # 端口被占用 / 无法绑定
        return False


def is_port_available(port: int) -> bool:
    """
    Windows 专用！能 100% 检测 Docker 容器占用的端口
    原理：不是 bind，而是尝试 connect
    🎯 为什么这个能行？
        旧方法（bind）：检查 Windows 自己是否占用端口
        Docker 不在 Windows 上占用 → 检测失败
        新方法（connect）：尝试真正访问 localhost:端口
        只要能访问 → 端口被占用（Docker 也会响应）
    """
    import socket

    try:
        # 尝试连接 127.0.0.1:端口
        # 如果能连接 = 端口被占用（包括Docker）
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)  # 超短超时
            result = s.connect_ex(('127.0.0.1', port))

        # 返回0 = 连接成功 = 端口被占用
        if result == 0:
            return False
        return True

    except Exception:
        return False


def is_instance_healthy(url: str) -> bool:
    """检查实例健康状态"""
    try:
        # 使用CoPaw的健康检查端点
        response = requests.get(
            f"{url}/api/version",
            timeout=5
        )
        return response.status_code == 200
    except requests.exceptions.RequestException:
        # 如果健康检查端点不存在，尝试基本连接
        try:
            response = requests.get(f"{url}/", timeout=3)
            return response.status_code in [200, 302]
        except:
            return False


if __name__ == '__main__':
    # print(is_port_available(9001))
    # print(is_instance_healthy("http://127.0.0.1:60393"))
    print(type(float(str(time.time()))))
