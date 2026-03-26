#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2026/3/25 11:35
# @Author  : gaohuan
# @Email   : 
# @FileName: postgres_client_pg8000.py
# @Desc    :
# manager/database/postgres_client_pg8000.py
import pg8000
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class PostgresClient:
    """PostgreSQL客户端，使用pg8000纯Python实现"""

    def __init__(self, host: str = "localhost", port: int = 5432,
                 database: str = "osclaw_manager", user: str = "postgres",
                 password: str = "postgres"):
        self.conn_params = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password,
            'timeout': 20
        }
        self._init_tables()

    def _get_connection(self):
        """获取数据库连接"""
        try:
            # pg8000自动处理UTF8编码
            conn = pg8000.connect(**self.conn_params)
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    def _init_tables(self):
        """初始化数据库表"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 用户表
            cursor.execute("""  
                CREATE TABLE IF NOT EXISTS users (  
                    user_id VARCHAR(255) PRIMARY KEY,  
                    user_name VARCHAR(255) NOT NULL,  
                    email VARCHAR(255),  
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP  
                )  
            """)

            # 实例表
            cursor.execute("""  
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
            cursor.execute("""  
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
            cursor.execute("""  
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
            cursor = conn.cursor()
            cursor.execute("""  
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
            cursor = conn.cursor()
            cursor.execute("""  
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
            cursor = conn.cursor()
            cursor.execute("""  
                SELECT * FROM instances WHERE user_id = %s  
            """, (user_id,))
            result = cursor.fetchone()
            if result:
                columns = ['user_id', 'container_id', 'container_name', 'port', 'status', 'config', 'created_at',
                           'updated_at']
                return dict(zip(columns, result))
            return None

    def list_instance(self) -> list[Dict]:
        """获取实例信息"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM instances")
            result = cursor.fetchall()
            res = []
            for item in result:
                columns = ['user_id', 'container_id', 'container_name', 'port', 'status', 'config', 'created_at',
                           'updated_at']
                res.append(dict(zip(columns, item)))
            return res

    def delete_instance(self, user_id: str) -> Optional[Dict]:
        """删除指定用户的实例信息"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""  
                DELETE FROM instances WHERE user_id = %s  
            """, (user_id,))
            conn.commit()

    def log_session(self, user_id: str, action: str, details: Dict = None,
                    ip_address: str = None, user_agent: str = None):
        """记录会话日志"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""  
                INSERT INTO session_logs (user_id, action, details, ip_address, user_agent)  
                VALUES (%s, %s, %s, %s, %s)  
            """, (user_id, action, json.dumps(details), ip_address, user_agent))
            conn.commit()

    def list_backups(self, user_id: str) -> List[Dict]:
        """列出用户备份"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""  
                SELECT * FROM backups WHERE user_id = %s ORDER BY created_at DESC  
            """, (user_id,))
            results = cursor.fetchall()
            columns = ['id', 'user_id', 'backup_path', 'backup_size', 'created_at']
            return [dict(zip(columns, result)) for result in results]
