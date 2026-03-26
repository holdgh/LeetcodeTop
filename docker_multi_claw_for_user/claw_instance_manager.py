#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time    : 2026/3/25 8:47
# @Author  : gaohuan
# @Email   : 
# @FileName: claw_instance_manager.py
# @Desc    :
import docker
import json
import time
import logging
from typing import Dict, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class CoPawInstanceManager:
    """CoPaw实例管理器 - 负责创建和管理用户专属的CoPaw容器实例"""

    def __init__(self, docker_client: docker.DockerClient = None):
        self.client = docker_client or docker.from_env()
        self.port_range = (9001, 9999)
        self.used_ports = set()
        self.instances = {}  # user_id -> instance_info

    def create_user_instance(self, user_id: str, user_name: str, config: Dict = None) -> Dict:
        """为用户创建独立的CoPaw实例"""
        try:
            # 分配端口
            port = self._allocate_port()
            container_name = f"copaw-user-{user_id}"
            data_volume = f"copaw-data-{user_id}"
            secrets_volume = f"copaw-secrets-{user_id}"

            # 创建数据卷
            self._create_volumes(data_volume, secrets_volume)

            # 准备环境变量
            environment = self._prepare_environment(user_id, user_name, config)

            # 启动容器
            container = self.client.containers.run(
                "osclaw:v010",
                name=container_name,
                ports={'8088/tcp': port},
                volumes={
                    data_volume: {'bind': '/app/working', 'mode': 'rw'},
                    secrets_volume: {'bind': '/app/working.secret', 'mode': 'rw'}
                },
                environment=environment,
                detach=True,
                restart_policy={"Name": "unless-stopped"},
                mem_limit='1g',  # 内存限制
                cpu_quota=50000  # CPU限制 (50%)
            )

            # 等待容器启动并初始化
            instance_info = self._initialize_instance(container, user_id, user_name, port)

            # 记录实例信息
            self.instances[user_id] = instance_info

            logger.info(f"Created CoPaw instance for user {user_id} on port {port}")
            return instance_info

        except Exception as e:
            logger.error(f"Failed to create instance for user {user_id}: {e}")
            raise

    def _allocate_port(self) -> int:
        """动态分配可用端口"""
        for port in range(self.port_range[0], self.port_range[1]):
            if port not in self.used_ports:
                # 检查端口是否真的可用
                if self._is_port_available(port):
                    self.used_ports.add(port)
                    return port
        raise Exception("No available ports in range")

    # def _is_port_available(self, port: int) -> bool:
    #     """
    #     检查端口是否可用（Windows/Linux 通用，真正准确）
    #     """
    #     import socket
    #
    #     try:
    #         # 测试 0.0.0.0（全网卡），这是服务真正使用的绑定方式
    #         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    #             s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    #             s.bind(('0.0.0.0', port))
    #         return True
    #
    #     except PermissionError:
    #         # 权限不足（Windows 常见于 <1024 端口）
    #         return False
    #
    #     except OSError:
    #         # 端口被占用 / 无法绑定
    #         return False

    def _is_port_available(self, port: int) -> bool:
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

    # def _is_port_available(self, port: int) -> bool:
    #     """检查端口是否可用"""
    #     import socket
    #     try:
    #         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    #             s.bind(('127.0.0.1', port))
    #             return True
    #     except:
    #         return False

    def _create_volumes(self, data_volume: str, secrets_volume: str):
        """创建数据卷"""
        try:
            self.client.volumes.create(data_volume)
            self.client.volumes.create(secrets_volume)
        except docker.errors.APIError as e:
            if "already exists" not in str(e):
                raise

    def _prepare_environment(self, user_id: str, user_name: str, config: Dict = None) -> Dict:
        """准备容器环境变量"""
        environment = {
            'COPAW_WORKING_DIR': '/app/working',
            # 'COPAW_AUTH_ENABLED': 'true',
            'COPAW_AUTH_ENABLED': 'false',  # 关闭登录页
            'COPAW_AUTH_USERNAME': user_name,
            'COPAW_AUTH_PASSWORD': f"{user_id}_{int(time.time())}",  # 生成临时密码
            'USER_ID': user_id,
            'USER_NAME': user_name,
            'COPAW_LOG_LEVEL': 'info'
        }

        # 添加自定义配置
        if config:
            environment.update(config.get('environment', {}))

        return environment

    def _initialize_instance(self, container, user_id: str, user_name: str, port: int) -> Dict:
        """初始化CoPaw实例"""
        # 等待容器启动
        time.sleep(5)
        container.reload()

        # 执行初始化命令
        exit_code, output = container.exec_run(
            "copaw init --defaults --accept-security",
            workdir="/app"
        )

        if exit_code != 0:
            raise Exception(f"Failed to initialize CoPaw: {output}")

        # 配置用户专属设置 采用默认配置
        # self._configure_user_instance(container, user_id, user_name)

        return {
            'user_id': user_id,
            'container_id': container.id,
            'container_name': container.name,
            'port': port,
            'url': f'http://127.0.0.1:{port}',
            'status': 'running',
            'created_at': time.time()
        }

    def _configure_user_instance(self, container, user_id: str, user_name: str):
        """配置用户专属的CoPaw设置"""
        # 基于CoPaw的agent.json结构配置用户专属设置
        agent_config = {
            "agent": {
                "name": f"{user_name}的AI助手",
                "description": f"为用户{user_name}定制的个人AI助手"
            },
            "channels": {
                "console": {
                    "enabled": True,
                    "bot_prefix": f"[{user_name}] "
                }
            },
            "heartbeat": {
                "enabled": False  # 默认关闭心跳以节省资源
            },
            "security": {
                "tool_guard": {"enabled": True},
                "file_guard": {"enabled": True}
            }
        }

        # 写入配置文件
        config_json = json.dumps(agent_config, ensure_ascii=False, indent=2)
        container.exec_run(
            f"echo '{config_json}' > /app/working/workspaces/default/agent.json",
            workdir="/app"
        )

        logger.info(f"Configured CoPaw instance for user {user_id}")

    def get_instance(self, user_id: str) -> Optional[Dict]:
        """获取用户实例信息"""
        return self.instances.get(user_id)

    def remove_instance(self, user_id: str) -> bool:
        """移除用户实例"""
        instance = self.instances.get(user_id)
        if not instance:
            return False

        try:
            # 停止并删除容器
            container = self.client.containers.get(instance['container_id'])
            container.stop()
            container.remove()

            # 释放端口
            self.used_ports.discard(instance['port'])

            # 从记录中移除
            del self.instances[user_id]

            logger.info(f"Removed CoPaw instance for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to remove instance for user {user_id}: {e}")
            return False

    def list_instances(self) -> List[Dict]:
        """列出所有实例"""
        return list(self.instances.values())