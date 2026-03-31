#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2026/3/25 9:48
# @Author  : gaohuan
# @Email   : 
# @FileName: resource_monitor.py
# @Desc    :
import time
import logging
import threading
from typing import Dict, List
from datetime import datetime, timedelta

from docker_multi_claw_for_user.claw_instance_manager import CoPawInstanceManager

logger = logging.getLogger(__name__)


class CoPawResourceMonitor:
    """CoPaw实例资源监控器"""

    def __init__(self, instance_manager: CoPawInstanceManager):
        self.instance_manager = instance_manager
        self.metrics = {}  # user_id -> metrics_history
        self.alerts = {}  # user_id -> alert_list
        self.monitoring = False
        self.monitor_thread = None

        # 监控阈值
        self.thresholds = {
            'memory_usage': 1024 * 1024 * 1024,  # 1GB
            'cpu_usage': 80.0,  # 80%
            'disk_usage': 90.0,  # 90%
        }

    def start_monitoring(self, interval: int = 60):
        """启动资源监控"""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Resource monitoring started")

    def stop_monitoring(self):
        """停止资源监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Resource monitoring stopped")

    def _monitor_loop(self, interval: int):
        """监控循环"""
        while self.monitoring:
            try:
                self._collect_all_metrics()
                self._check_thresholds()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")

    def _collect_all_metrics(self):
        """收集所有实例的指标"""
        instances = self.instance_manager.list_instances()

        for instance in instances:
            user_id = instance['user_id']
            try:
                metrics = self._collect_instance_metrics(instance)

                # 存储指标历史
                if user_id not in self.metrics:
                    self.metrics[user_id] = []

                self.metrics[user_id].append({
                    'timestamp': time.time(),
                    **metrics
                })

                # 保留最近1小时的数据
                cutoff_time = time.time() - 3600
                self.metrics[user_id] = [
                    m for m in self.metrics[user_id]
                    if m['timestamp'] > cutoff_time
                ]

            except Exception as e:
                logger.error(f"Failed to collect metrics for {user_id}: {e}")

    # def _collect_instance_metrics(self, instance: Dict) -> Dict:
    #     """收集单个实例的指标"""
    #     container_id = instance['container_id']
    #     container = self.instance_manager.client.containers.get(container_id)
    #
    #     # 获取容器统计信息
    #     stats = container.stats(stream=False)
    #     print(f"容器统计信息：{stats}")
    #     # 计算CPU使用率
    #     cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - stats['precpu_stats']['cpu_usage']['total_usage']
    #     system_cpu_delta = stats['cpu_stats']['system_cpu_usage'] - stats['precpu_stats']['system_cpu_usage']
    #     cpu_usage = (cpu_delta / system_cpu_delta) * len(
    #         stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100 if system_cpu_delta > 0 else 0
    #
    #     # 内存使用
    #     memory_usage = stats['memory_stats']['usage']
    #     memory_limit = stats['memory_stats']['limit']
    #     memory_percent = (memory_usage / memory_limit) * 100
    #
    #     # 网络IO
    #     network_io = {}
    #     for interface, data in stats.get('networks', {}).items():
    #         network_io[interface] = {
    #             'rx_bytes': data['rx_bytes'],
    #             'tx_bytes': data['tx_bytes']
    #         }
    #
    #     return {
    #         'cpu_usage': cpu_usage,
    #         'memory_usage': memory_usage,
    #         'memory_percent': memory_percent,
    #         'network_io': network_io,
    #         'container_status': container.status
    #     }

    def _collect_instance_metrics(self, instance: Dict) -> Dict:
        """收集单个实例的指标（兼容空数据 + 完整Docker stats）"""
        container_id = instance['container_id']
        container = self.instance_manager.client.containers.get(container_id)

        stats = container.stats(stream=False)

        # ===================== CPU 使用率 =====================
        cpu_usage = 0.0
        try:
            cpu_stats = stats.get("cpu_stats", {})
            precpu_stats = stats.get("precpu_stats", {})

            cpu_total = cpu_stats.get("cpu_usage", {}).get("total_usage", 0)
            pre_cpu_total = precpu_stats.get("cpu_usage", {}).get("total_usage", 0)
            cpu_delta = cpu_total - pre_cpu_total

            system_cpu = cpu_stats.get("system_cpu_usage", 0)
            pre_system_cpu = precpu_stats.get("system_cpu_usage", 0)
            system_delta = system_cpu - pre_system_cpu

            online_cpus = cpu_stats.get("online_cpus", 1)

            if system_delta > 0 and cpu_delta > 0:
                cpu_usage = (cpu_delta / system_delta) * online_cpus * 100
        except Exception:
            cpu_usage = 0.0

        # ===================== 内存 =====================
        memory_usage = 0
        memory_percent = 0.0
        try:
            mem = stats.get("memory_stats", {})
            usage = mem.get("usage", 0)
            limit = mem.get("limit", 1)
            memory_usage = usage
            if limit > 0:
                memory_percent = (usage / limit) * 100
        except Exception:
            memory_usage = 0
            memory_percent = 0.0

        # ===================== 网络 =====================
        network_io = {}
        try:
            networks = stats.get("networks", {})
            for iface, data in networks.items():
                network_io[iface] = {
                    "rx_bytes": data.get("rx_bytes", 0),
                    "tx_bytes": data.get("tx_bytes", 0)
                }
        except Exception:
            network_io = {}

        # ===================== 容器状态 =====================
        container_status = container.status

        return {
            "cpu_usage": round(cpu_usage, 2),
            "memory_usage": memory_usage,
            "memory_percent": round(memory_percent, 2),
            "network_io": network_io,
            "container_status": container_status
        }

    def _check_thresholds(self):
        """检查阈值并发送告警"""
        current_time = time.time()

        for user_id, metrics_history in self.metrics.items():
            if not metrics_history:
                continue

            latest_metrics = metrics_history[-1]

            # 检查内存使用
            if latest_metrics['memory_usage'] > self.thresholds['memory_usage']:
                self._send_alert(user_id, "High memory usage", latest_metrics)

            # 检查CPU使用
            if latest_metrics['cpu_usage'] > self.thresholds['cpu_usage']:
                self._send_alert(user_id, "High CPU usage", latest_metrics)

    def _send_alert(self, user_id: str, message: str, metrics: Dict):
        """发送告警"""
        alert = {
            'timestamp': time.time(),
            'message': message,
            'metrics': metrics
        }

        if user_id not in self.alerts:
            self.alerts[user_id] = []

        self.alerts[user_id].append(alert)

        # 保留最近100条告警
        self.alerts[user_id] = self.alerts[user_id][-100:]

        logger.warning(f"Alert for user {user_id}: {message}")

    def get_metrics_summary(self, user_id: str) -> Dict:
        """获取用户指标摘要"""
        if user_id not in self.metrics or not self.metrics[user_id]:
            return {}

        metrics_history = self.metrics[user_id]
        latest = metrics_history[-1]

        # 计算平均值
        avg_cpu = sum(m['cpu_usage'] for m in metrics_history) / len(metrics_history)
        avg_memory = sum(m['memory_usage'] for m in metrics_history) / len(metrics_history)

        return {
            'current': latest,
            'averages': {
                'cpu_usage': avg_cpu,
                'memory_usage': avg_memory
            },
            'sample_count': len(metrics_history)
        }

    def cleanup_idle_instances(self, idle_threshold: int = 3600):
        """清理空闲实例"""
        current_time = time.time()

        for user_id, metrics_history in list(self.metrics.items()):
            if not metrics_history:
                continue

            last_activity = metrics_history[-1]['timestamp']
            if current_time - last_activity > idle_threshold:
                logger.info(f"Cleaning up idle instance for user {user_id}")
                # 这里可以调用生命周期管理器来停止实例
                # self.lifecycle_manager._stop_instance(user_id)