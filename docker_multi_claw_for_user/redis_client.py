#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import time

# @Time    : 2026/3/25 10:40
# @Author  : gaohuan
# @Email   : 
# @FileName: redis_client.py
# @Desc    :
# manager/database/redis_client.py
import redis
import json
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class RedisClient:
    """Redis客户端，用于缓存和会话管理"""

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.prefix = "osclaw:"

    def cache_user_instance(self, user_id: str, instance_info: Dict):
        """缓存用户实例信息"""
        key = f"{self.prefix}instance:{user_id}"
        self.client.setex(key, 3600, json.dumps(instance_info))  # 1小时过期

    def get_user_instance(self, user_id: str) -> Optional[Dict]:
        """获取缓存的用户实例信息"""
        key = f"{self.prefix}instance:{user_id}"
        data = self.client.get(key)
        return json.loads(data) if data else None

    def cache_user_session(self, user_id: str, session_data: Dict):
        """缓存用户会话"""
        key = f"{self.prefix}session:{user_id}"
        self.client.setex(key, 1800, json.dumps(session_data))  # 30分钟过期

    def set_user_active(self, user_id: str):
        """设置用户活跃状态"""
        key = f"{self.prefix}active:{user_id}"
        self.client.set(key, str(time.time()))  # 不设置过期时间，仅在用户主动删除实例或者用户退出后30分钟删除或者服务停止时删除

    def clean_user_active(self, user_id: str):
        """删除用户活跃状态，使用场景：用户主动删除实例或者用户退出后30分钟删除"""
        key = f"{self.prefix}active:{user_id}"
        self.client.delete(key)

    def clean_all(self):
        """删除用户活跃状态，使用场景：服务停止时清空所有缓存数据"""
        keys = self.client.keys(f"{self.prefix}*")
        # 2. 批量删除
        if keys:
            self.client.delete(*keys)

    def get_user_active(self, user_id: str) -> float:
        """检查用户是否活跃"""
        key = f"{self.prefix}active:{user_id}"
        active_start_time = self.client.get(key)
        if active_start_time:
            return float(active_start_time)
        return 0.0
