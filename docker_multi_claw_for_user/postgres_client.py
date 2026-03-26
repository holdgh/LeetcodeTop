#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2026/3/25 10:41
# @Author  : gaohuan
# @Email   : 
# @FileName: postgres_client.py
# @Desc    :
# manager/database/postgres_client.py
import psycopg2
import psycopg2.extras
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class PostgresClient:
    """PostgreSQL客户端，用于持久化数据存储"""

    def __init__(self, host: str = "localhost", port: int = 5432,
                 database: str = "osclaw_manager", user: str = "postgres",
                 password: str = "postgres"):
        self.conn_params = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password,
            'client_encoding': 'UTF8'
        }
        self._init_tables()

    def _get_connection(self):
        """获取数据库连接"""
        return psycopg2.connect(**self.conn_params)

    def _init_tables(self):
        """初始化数据库表"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # 用户表
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id VARCHAR(255) PRIMARY KEY,
                        user_name VARCHAR(255) NOT NULL,
                        email VARCHAR(255),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # 实例表
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS instances (
                        user_id VARCHAR(255) PRIMARY KEY,
                        container_id VARCHAR(255) NOT NULL,
                        container_name VARCHAR(255) NOT NULL,
                        port INTEGER NOT NULL,
                        status VARCHAR(50) DEFAULT 'stopped',
                        config JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users(user_id)
                    )
                """)

                # 备份记录表
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS backups (
                        id SERIAL PRIMARY KEY,
                        user_id VARCHAR(255) NOT NULL,
                        backup_path VARCHAR(500) NOT NULL,
                        backup_size BIGINT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users(user_id)
                    )
                """)

                # 会话日志表
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS session_logs (
                        id SERIAL PRIMARY KEY,
                        user_id VARCHAR(255) NOT NULL,
                        action VARCHAR(100) NOT NULL,
                        details JSONB,
                        ip_address INET,
                        user_agent TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users(user_id)
                    )
                """)

            conn.commit()

    def create_user(self, user_id: str, user_name: str, email: str = None):
        """创建用户"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO users (user_id, user_name, email)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (user_id) DO UPDATE SET
                        user_name = EXCLUDED.user_name,
                        email = EXCLUDED.email,
                        updated_at = CURRENT_TIMESTAMP
                """, (user_id, user_name, email))
            conn.commit()

    def save_instance(self, user_id: str, instance_info: Dict):
        """保存实例信息"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO instances (user_id, container_id, container_name, port, status, config)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id) DO UPDATE SET
                        container_id = EXCLUDED.container_id,
                        container_name = EXCLUDED.container_name,
                        port = EXCLUDED.port,
                        status = EXCLUDED.status,
                        config = EXCLUDED.config,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    user_id,
                    instance_info['container_id'],
                    instance_info['container_name'],
                    instance_info['port'],
                    instance_info['status'],
                    json.dumps(instance_info)
                ))
            conn.commit()

    def get_instance(self, user_id: str) -> Optional[Dict]:
        """获取实例信息"""
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM instances WHERE user_id = %s
                """, (user_id,))
                result = cur.fetchone()
                return dict(result) if result else None

    def log_session(self, user_id: str, action: str, details: Dict = None,
                    ip_address: str = None, user_agent: str = None):
        """记录会话日志"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO session_logs (user_id, action, details, ip_address, user_agent)
                    VALUES (%s, %s, %s, %s, %s)
                """, (user_id, action, json.dumps(details), ip_address, user_agent))
            conn.commit()

    def list_backups(self, user_id: str) -> List[Dict]:
        """列出用户备份"""
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM backups WHERE user_id = %s ORDER BY created_at DESC
                """, (user_id,))
                return [dict(row) for row in cur.fetchall()]