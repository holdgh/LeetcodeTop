#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2026/3/25 9:49
# @Author  : gaohuan
# @Email   : 
# @FileName: data_backup_recover.py
# @Desc    :
import tarfile
import gzip
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List

from docker_multi_claw_for_user.claw_instance_manager import CoPawInstanceManager
from docker_multi_claw_for_user.postgres_client_pg8000 import PostgresClient

logger = logging.getLogger(__name__)


class CoPawBackupManager:
    """CoPaw数据备份和恢复管理器"""

    def __init__(self, backup_dir: str = "/backup"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # 初始化PostgreSQL客户端
        self.postgres_client = PostgresClient()

    def backup_user_data(self, user_id: str, instance_manager: CoPawInstanceManager) -> str:
        """备份用户数据"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{user_id}_backup_{timestamp}.tar.gz"
            backup_path = self.backup_dir / backup_filename

            data_volume = f"copaw-data-{user_id}"
            secrets_volume = f"copaw-secrets-{user_id}"

            # 创建备份容器
            backup_container = instance_manager.client.containers.run(
                "alpine:latest",
                command=f"tar czf /backup/{backup_filename} -C /data . && tar czf /backup/secrets_{backup_filename} -C /secrets .",
                volumes={
                    data_volume: {'bind': '/data', 'mode': 'ro'},
                    secrets_volume: {'bind': '/secrets', 'mode': 'ro'},
                    str(self.backup_dir): {'bind': '/backup', 'mode': 'rw'}
                },
                remove=True
            )

            # 等待备份完成
            backup_container.wait()

            # 创建备份元数据
            metadata = {
                'user_id': user_id,
                'timestamp': timestamp,
                'backup_file': backup_filename,
                'secrets_file': f"secrets_{backup_filename}",
                'volumes': [data_volume, secrets_volume],
                'created_at': datetime.now().isoformat()
            }

            metadata_path = self.backup_dir / f"{user_id}_backup_{timestamp}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Backup completed for user {user_id}: {backup_filename}")
            # 记录备份到数据库
            backup_size = backup_path.stat().st_size if backup_path.exists() else 0
            with self.postgres_client._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                                INSERT INTO backups (user_id, backup_path, backup_size)
                                VALUES (%s, %s, %s)
                            """, (user_id, str(backup_path), backup_size))
                conn.commit()
            return str(backup_path)

        except Exception as e:
            logger.error(f"Backup failed for user {user_id}: {e}")
            raise

    def restore_user_data(self, user_id: str, backup_path: str, instance_manager: CoPawInstanceManager):
        """恢复用户数据"""
        try:
            backup_file = Path(backup_path)
            if not backup_file.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_path}")

            # 提取用户ID和时间戳
            parts = backup_file.stem.split('_')
            if len(parts) < 3:
                raise ValueError("Invalid backup filename format")

            restored_user_id = parts[0]
            timestamp = parts[2]

            data_volume = f"copaw-data-{restored_user_id}"
            secrets_volume = f"copaw-secrets-{restored_user_id}"

            # 创建数据卷（如果不存在）
            try:
                instance_manager.client.volumes.create(data_volume)
                instance_manager.client.volumes.create(secrets_volume)
            except Exception:
                pass  # 卷可能已存在

            # 恢复数据
            restore_container = instance_manager.client.containers.run(
                "alpine:latest",
                command=f"tar xzf /backup/{backup_file.name} -C /data && tar xzf /backup/secrets_{backup_file.name} -C /secrets",
                volumes={
                    data_volume: {'bind': '/data', 'mode': 'rw'},
                    secrets_volume: {'bind': '/secrets', 'mode': 'rw'},
                    str(self.backup_dir): {'bind': '/backup', 'mode': 'ro'}
                },
                remove=True
            )

            # 等待恢复完成
            restore_container.wait()

            logger.info(f"Restore completed for user {restored_user_id}")

        except Exception as e:
            logger.error(f"Restore failed for user {user_id}: {e}")
            raise

    def list_backups(self, user_id: str = None) -> List[Dict]:
        """列出备份文件"""
        """从数据库列出备份文件"""
        return self.postgres_client.list_backups(user_id)
        # backups = []
        #
        # for metadata_file in self.backup_dir.glob("*_backup_*.json"):
        #     try:
        #         with open(metadata_file, 'r') as f:
        #             metadata = json.load(f)
        #
        #         if user_id is None or metadata['user_id'] == user_id:
        #             backups.append(metadata)
        #     except Exception as e:
        #         logger.error(f"Failed to read metadata {metadata_file}: {e}")
        #
        # return sorted(backups, key=lambda x: x['created_at'], reverse=True)

    def delete_backup(self, user_id: str, timestamp: str):
        """删除备份文件"""
        try:
            backup_filename = f"{user_id}_backup_{timestamp}.tar.gz"
            secrets_filename = f"secrets_{user_id}_backup_{timestamp}.tar.gz"
            metadata_filename = f"{user_id}_backup_{timestamp}.json"

            # 删除文件
            for filename in [backup_filename, secrets_filename, metadata_filename]:
                file_path = self.backup_dir / filename
                if file_path.exists():
                    file_path.unlink()

            logger.info(f"Deleted backup for user {user_id}: {timestamp}")

        except Exception as e:
            logger.error(f"Failed to delete backup for user {user_id}: {e}")
            raise

    def cleanup_old_backups(self, user_id: str, keep_count: int = 5):
        """清理旧备份，保留最新的几个"""
        backups = self.list_backups(user_id)

        if len(backups) <= keep_count:
            return

        # 删除最旧的备份
        old_backups = backups[keep_count:]
        for backup in old_backups:
            self.delete_backup(backup['user_id'], backup['timestamp'])

        logger.info(f"Cleaned up {len(old_backups)} old backups for user {user_id}")