#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time    : 2026/3/25 9:33
# @Author  : gaohuan
# @Email   : 
# @FileName: claw_lifespan_manager.py
# @Desc    :
import asyncio
import time
import requests
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
from docker_multi_claw_for_user.claw_instance_manager import CoPawInstanceManager
from docker_multi_claw_for_user.postgres_client_pg8000 import PostgresClient
# from docker_multi_claw_for_user.redis_client import RedisClient

logger = logging.getLogger(__name__)


class CoPawLifecycleManager:
    """CoPaw实例生命周期管理器"""

    def __init__(self, instance_manager: CoPawInstanceManager):
        self.instance_manager = instance_manager
        self.load_balancer = None  # 将在初始化时设置

        # 初始化数据库连接
        # self.redis_client = RedisClient()
        self.postgres_client = PostgresClient()

    def set_load_balancer(self, load_balancer):
        """设置负载均衡器"""
        self.load_balancer = load_balancer

    def on_user_register(self, user_id: str, user_name: str, config: Dict = None) -> Dict:
        """用户注册时创建实例"""
        """
        逻辑如下：
            1、新增用户信息至数据库
            2、创建osclaw实例并新增该实例信息至数据库
            3、更新负载均衡器【实例启动--生成nginx配置--写入文件--重启nginx】【看起来没啥用】
        """
        try:
            # 保存用户信息到数据库
            self.postgres_client.create_user(user_id, user_name)
            # 创建实例
            instance_info = self.instance_manager.create_user_instance(user_id, user_name, config)
            # 将用户活跃时间记录到缓存
            # self.redis_client.set_user_active(user_id)
            # 保存实例信息到数据库
            self.postgres_client.save_instance(user_id, instance_info)

            # 缓存到Redis TODO 为何要放redis一份数据呢？
            # self.redis_client.cache_user_instance(user_id, instance_info)
            # 更新负载均衡器
            if self.load_balancer:
                self._update_load_balancer(instance_info)
            # 记录日志
            self.postgres_client.log_session(user_id, "register", instance_info)
            logger.info(f"User registered: {user_id}, instance created")
            return instance_info

        except Exception as e:
            logger.error(f"Failed to handle user registration for {user_id}: {e}")
            # 异步创建或重试 TODO 关闭重试机制
            # self._schedule_instance_creation(user_id, user_name, config)
            return {'status': 'scheduled', 'user_id': user_id}

    def on_user_login(self, user_id: str) -> Optional[str]:
        """用户登录时获取实例URL"""
        # 更新活跃时间
        # self.redis_client.set_user_active(user_id)

        # 从数据库获取
        instance = self.postgres_client.get_instance(user_id)
        # 先从缓存获取实例
        # instance = self.redis_client.get_user_instance(user_id)
        # if not instance:
        #     # 从数据库获取
        #     instance = self.postgres_client.get_instance(user_id)
        #     if instance:
        #         self.redis_client.cache_user_instance(user_id, instance)
        if not instance:
            # 实例不存在，重新创建
            return self._recreate_instance(user_id)

        # 检查实例健康状态
        if not self._is_instance_healthy(instance):
            logger.warning(f"Instance unhealthy for user {user_id}, attempting restart")
            return self._restart_instance(user_id)
        # 记录登录日志
        self.postgres_client.log_session(user_id, "login", {"instance_url": instance['url']})
        return instance['url']

    def on_user_logout(self, user_id: str):
        """用户登出时的处理"""
        self._stop_instance(user_id)

        # # 检查是否应该停止实例
        # if self._should_stop_instance(user_id):
        #     self._stop_instance(user_id)

    def on_user_delete(self, user_id: str):
        """用户删除时清理所有资源"""
        logger.info(f"Cleaning up all resources for user {user_id}")
        self._cleanup_instance(user_id)

    def _update_load_balancer(self, instance_info: Dict):
        """更新负载均衡器配置"""
        if not self.load_balancer:
            return

        backend_config = {
            'server': f"127.0.0.1:{instance_info['port']}",
            'weight': 1,
            'max_fails': 3,
            'fail_timeout': 30,
            'user_id': instance_info['user_id']
        }

        try:
            self.load_balancer.add_backend(instance_info['user_id'], backend_config)
            logger.info(f"Added backend to load balancer for user {instance_info['user_id']}")
        except Exception as e:
            logger.error(f"Failed to update load balancer: {e}")

    def _recreate_instance(self, user_id: str) -> Optional[str]:
        """重新创建用户实例"""
        logger.info(f"Recreating instance for user {user_id}")

        # 清理旧实例
        self._cleanup_instance(user_id)

        # 获取用户信息
        user_info = self._get_user_info(user_id)
        if not user_info:
            logger.error(f"User info not found for {user_id}")
            return None

        # 创建新实例
        try:
            instance_info = self.instance_manager.create_user_instance(
                user_id, user_info['name'], user_info.get('config')
            )
            # 保存实例信息到数据库
            self.postgres_client.save_instance(user_id, instance_info)
            self._update_load_balancer(instance_info)

            return instance_info['url']
        except Exception as e:
            logger.error(f"Failed to recreate instance for {user_id}: {e}")
            return None

    def _is_instance_healthy(self, instance_info: Dict) -> bool:
        """检查实例健康状态"""
        try:
            # 使用CoPaw的健康检查端点
            response = requests.get(
                f"{instance_info['url']}/api/version",  # 调用版本接口，判断实例是否正常存活
                timeout=5
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            # 如果健康检查端点不存在，尝试基本连接
            try:
                response = requests.get(f"{instance_info['url']}/", timeout=3)
                return response.status_code in [200, 302]
            except:
                return False

    def _restart_instance(self, user_id: str) -> Optional[str]:
        """重启用户实例"""
        # 从数据库获取
        instance = self.postgres_client.get_instance(user_id)
        if not instance:
            return self._recreate_instance(user_id)

        try:
            # 重启容器
            container = self.instance_manager.client.containers.get(instance['container_id'])
            container.restart()

            # 等待容器启动
            time.sleep(5)

            # 验证健康状态
            if self._is_instance_healthy(instance):
                logger.info(f"Successfully restarted instance for user {user_id}")
                return instance['url']
            else:
                logger.warning(f"Instance still unhealthy after restart for user {user_id}")
                return self._recreate_instance(user_id)

        except Exception as e:
            logger.error(f"Failed to restart instance for user {user_id}: {e}")
            return self._recreate_instance(user_id)

    # def _should_stop_instance(self, user_id: str) -> bool:  # TODO 暂时弃用，思想可以采用【最后一次活跃时间距今超过半小时，则自动清理实例】
    #     """判断是否应该停止实例（基于用户活跃度）"""
    #     last_active = self.redis_client.get_user_active(user_id)
    #     if last_active < 1.0:
    #         return True
    #     idle_threshold = 30 * 60  # 30分钟
    #
    #     return (time.time() - last_active) > idle_threshold

    def _stop_instance(self, user_id: str):
        """停止用户实例"""
        # 从数据库获取
        instance = self.postgres_client.get_instance(user_id)
        if not instance:
            return

        try:
            # 停止容器
            container = self.instance_manager.client.containers.get(instance['container_id'])
            container.stop()

            # 从负载均衡器移除
            if self.load_balancer:
                self.load_balancer.remove_backend(user_id)

            # 更新状态
            instance['status'] = 'stopped'
            logger.info(f"Stopped instance for user {user_id}")

        except Exception as e:
            logger.error(f"Failed to stop instance for user {user_id}: {e}")

    def _cleanup_instance(self, user_id: str):
        """完全清理用户实例和数据"""
        # 从数据库获取
        instance = self.postgres_client.get_instance(user_id)
        if not instance:
            return

        try:
            # 从负载均衡器移除
            if self.load_balancer:
                self.load_balancer.remove_backend(user_id)

            # 移除实例
            self.instance_manager.remove_instance(user_id)

            # 从数据库删除实例
            self.postgres_client.delete_instance(user_id)

            logger.info(f"Completely cleaned up instance for user {user_id}")

        except Exception as e:
            logger.error(f"Failed to cleanup instance for user {user_id}: {e}")

    def _schedule_instance_creation(self, user_id: str, user_name: str, config: Dict):
        """调度实例创建（异步重试）"""

        def delayed_creation():
            time.sleep(10)  # 等待10秒后重试
            try:
                self.on_user_register(user_id, user_name, config)
            except Exception as e:
                logger.error(f"Delayed instance creation failed for {user_id}: {e}")

        # 在后台线程中执行
        import threading
        thread = threading.Thread(target=delayed_creation)
        thread.daemon = True
        thread.start()

    def _get_user_info(self, user_id: str) -> Optional[Dict]:
        """获取用户信息（需要根据实际数据库实现）"""
        # 这里应该从数据库获取用户信息
        # 示例实现
        return {
            'name': f"User_{user_id}",
            'email': f"user_{user_id}@example.com",
            'config': {}
        }

    def start_health_monitoring(self):
        """启动健康监控任务"""

        def monitor():
            while True:
                try:
                    self._check_all_instances_health()
                    time.sleep(60)  # 每分钟检查一次
                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")

        import threading
        monitor_thread = threading.Thread(target=monitor)
        monitor_thread.daemon = True
        monitor_thread.start()
        logger.info("Health monitoring started")

    def _check_all_instances_health(self):
        """检查所有实例的健康状态"""
        # TODO 查询数据库所有实例列表
        instance_list = self.postgres_client.list_instance()
        for instance in instance_list:
            if instance['status'] == 'running':
                if not self._is_instance_healthy(instance):
                    logger.warning(f"Unhealthy instance detected for user {instance['user_id']}")
                    # 尝试重启
                    self._restart_instance(instance['user_id'])