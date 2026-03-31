#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# TODO 暂时弃用
# @Time    : 2026/3/25 9:47
# @Author  : gaohuan
# @Email   : 
# @FileName: load_banlancer.py
# @Desc    :
import json
import logging
from typing import Dict, List
from pathlib import Path
import subprocess

logger = logging.getLogger(__name__)


class LoadBalancer:
    """简单的负载均衡器实现（基于Nginx配置）"""

    # def __init__(self, config_path: str = "/etc/nginx/conf.d/copaw.conf"):
    def __init__(self, config_path: str = r"C:\Users\gaohu\software\nginx-1.28.3\conf\osclaw.conf"):
        self.config_path = Path(config_path)
        self.backends = {}  # user_id -> backend_config
        self.nginx_reload_cmd = ["nginx", "-s", "reload"]

    def add_backend(self, user_id: str, backend_config: Dict):
        """添加后端服务器"""
        self.backends[user_id] = backend_config
        self._update_config()
        self._reload_nginx()
        logger.info(f"Added backend for user {user_id}: {backend_config['server']}")

    def remove_backend(self, user_id: str):
        """移除后端服务器"""
        if user_id in self.backends:
            del self.backends[user_id]
            self._update_config()
            self._reload_nginx()
            logger.info(f"Removed backend for user {user_id}")

    def _update_config(self):
        """更新Nginx配置文件"""
        config_content = self._generate_nginx_config()

        # 确保配置目录存在
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # 写入配置文件
        with open(self.config_path, 'w') as f:
            f.write(config_content)

    def _generate_nginx_config(self) -> str:
        """生成Nginx配置内容"""
        upstream_servers = []
        for user_id, backend in self.backends.items():
            upstream_servers.append(
                f"    server {backend['server']} weight={backend['weight']} max_fails={backend['max_fails']} fail_timeout={backend['fail_timeout']};")

        config = f"""
upstream osclaw_backend {{
{chr(10).join(upstream_servers)}
}}

server {{
    listen 80;

    # 根据用户ID路由到对应实例
    location /api/ {{
        proxy_set_header X-User-ID $http_x_user_id;
        proxy_pass http://osclaw_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # 超时设置
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }}

    # 健康检查端点
    location /health {{
        access_log off;
        return 200 "healthy\\n";
        add_header Content-Type text/plain;
    }}
}}
"""
        return config

    def _reload_nginx(self):
        """重新加载Nginx配置"""
        try:
            subprocess.run(self.nginx_reload_cmd, check=True, capture_output=True)
            logger.info("Nginx configuration reloaded successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to reload nginx: {e}")

    def get_backend_status(self) -> Dict:
        """获取所有后端状态"""
        return {
            user_id: {
                'server': backend['server'],
                'weight': backend['weight'],
                'status': 'active'
            }
            for user_id, backend in self.backends.items()
        }