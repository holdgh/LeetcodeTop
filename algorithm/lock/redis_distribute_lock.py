#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2026/3/19 16:16
# @Author  : gaohuan
# @Email   : 
# @FileName: redis_distribute_lock.py
# @Desc    :
import redis
import uuid
import threading
import time
from typing import Optional, Any


class ReentrantRedisLock:
    """
    可重入Redis分布式锁（修复：每次重入同步Redis计数）
    核心原则：每次加锁/解锁都必须同步Redis，Redis是唯一权威数据源
    """

    def __init__(
            self,
            redis_client: redis.Redis,
            lock_key: str,
            expire_seconds: int = 30,
            retry_interval: float = 0.1,
            acquire_timeout: Optional[float] = None,
    ):
        if expire_seconds <= 0:
            raise ValueError("锁过期时间必须大于0秒")
        if retry_interval <= 0:
            raise ValueError("重试间隔必须大于0秒")

        self.redis = redis_client
        self.lock_key = lock_key
        self.expire = expire_seconds
        self.retry_interval = retry_interval
        self.acquire_timeout = acquire_timeout

        # 客户端+线程唯一标识
        self.client_id = f"{uuid.uuid4()}_{threading.get_ident()}"
        # 本地计数（仅为了快速判断是否持有锁，不替代Redis）
        self.reentrant_count = 0
        self._local_lock = threading.Lock()

        # 看门狗
        self._watchdog_thread: Optional[threading.Thread] = None
        self._watchdog_stop = threading.Event()

        # 预加载Lua脚本
        self._load_lua_scripts()

    def _load_lua_scripts(self) -> None:
        """Lua脚本：每次加锁都更新Redis计数，每次解锁都减少Redis计数"""
        # 1. 加锁脚本（返回最新重入次数）
        self._lock_script = self.redis.register_script("""
            local lock_key = KEYS[1]
            local client_id = ARGV[1]
            local expire = ARGV[2]

            -- 锁不存在：初始化计数=1
            if redis.call('EXISTS', lock_key) == 0 then
                redis.call('HSET', lock_key, client_id, 1)
                redis.call('EXPIRE', lock_key, expire)
                return 1
            end

            -- 锁属于当前客户端：计数+1
            if redis.call('HEXISTS', lock_key, client_id) == 1 then
                local new_count = redis.call('HINCRBY', lock_key, client_id, 1)
                redis.call('EXPIRE', lock_key, expire)
                return new_count
            end

            -- 锁被其他客户端持有：加锁失败
            return 0
        """)

        # 2. 解锁脚本（返回剩余计数）
        self._unlock_script = self.redis.register_script("""
            local lock_key = KEYS[1]
            local client_id = ARGV[1]
            local expire = ARGV[2]

            -- 锁不属于当前客户端：返回-1
            if redis.call('HEXISTS', lock_key, client_id) == 0 then
                return -1
            end

            -- 计数-1
            local remain_count = redis.call('HINCRBY', lock_key, client_id, -1)
            if remain_count > 0 then
                redis.call('EXPIRE', lock_key, expire)
                return remain_count  -- 计数>0，返回剩余次数
            end

            -- 计数=0，删除锁
            redis.call('DEL', lock_key)
            return 0
        """)

        # 3. 续期脚本
        self._renew_script = self.redis.register_script("""
            local lock_key = KEYS[1]
            local client_id = ARGV[1]
            local expire = ARGV[2]
            if redis.call('HEXISTS', lock_key, client_id) == 1 then
                redis.call('EXPIRE', lock_key, expire)
                return 1
            end
            return 0
        """)

    def _start_watchdog(self) -> None:
        """启动看门狗"""
        self._watchdog_stop.clear()

        def watchdog_worker():
            while not self._watchdog_stop.is_set():
                time.sleep(self.expire / 3)
                self._renew_script(keys=[self.lock_key], args=[self.client_id, self.expire])

        self._watchdog_thread = threading.Thread(target=watchdog_worker, daemon=True)
        self._watchdog_thread.start()

    def _stop_watchdog(self) -> None:
        """停止看门狗"""
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            self._watchdog_stop.set()
            self._watchdog_thread.join(timeout=1)
        self._watchdog_thread = None

    def acquire(self, blocking: bool = True) -> bool:
        """
        加锁：每次重入都调用Lua脚本更新Redis计数
        彻底放弃“本地仅加计数”的优化，确保Redis计数准确
        """
        with self._local_lock:
            start_time = time.time()
            while True:
                # 每次加锁都执行Lua脚本（无论是否本地持有锁）
                redis_count = self._lock_script(
                    keys=[self.lock_key],
                    args=[self.client_id, self.expire]
                )

                if redis_count > 0:
                    # 加锁成功，同步Redis计数到本地
                    self.reentrant_count = redis_count
                    # 首次加锁启动看门狗
                    if redis_count == 1:
                        self._start_watchdog()
                    return True

                # 非阻塞模式失败
                if not blocking:
                    return False

                # 超时判断
                if self.acquire_timeout and (time.time() - start_time) > self.acquire_timeout:
                    return False

                time.sleep(self.retry_interval)

    def release(self) -> bool:
        """
        解锁：每次解锁都调用Lua脚本更新Redis计数
        """
        with self._local_lock:
            # 本地无锁，直接返回失败
            if self.reentrant_count == 0:
                return False

            # 执行Redis解锁脚本
            redis_remain_count = self._unlock_script(
                keys=[self.lock_key],
                args=[self.client_id, self.expire]
            )

            # 处理返回结果
            if redis_remain_count == -1:
                # 锁不属于当前客户端，重置本地计数
                self.reentrant_count = 0
                return False
            elif redis_remain_count == 0:
                # 锁已删除，重置本地计数+停止看门狗
                self.reentrant_count = 0
                self._stop_watchdog()
            else:
                # 仍有重入次数，同步本地计数
                self.reentrant_count = redis_remain_count

            return True

    def get_redis_count(self) -> int:
        """获取Redis中的真实重入次数"""
        count = self.redis.hget(self.lock_key, self.client_id)
        return int(count) if count else 0

    def __enter__(self) -> "ReentrantRedisLock":
        if not self.acquire(blocking=True):
            raise TimeoutError(f"获取锁[{self.lock_key}]超时")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()

    def __del__(self) -> None:
        with self._local_lock:
            if self.reentrant_count > 0:
                self.redis.delete(self.lock_key)
                self.reentrant_count = 0
            self._stop_watchdog()


# ------------------------------ 验证测试 ------------------------------
if __name__ == "__main__":
    # 初始化Redis
    redis_client = redis.Redis(
        host="127.0.0.1",
        port=6379,
        db=0,
        decode_responses=True,
        password=None
    )

    # 创建锁
    lock = ReentrantRedisLock(redis_client, "lock:test", expire_seconds=30)

    # 测试重入计数同步
    print("=== 测试重入计数 ===")
    # 第一次加锁
    lock.acquire()
    print(f"第一次加锁 - 本地计数：{lock.reentrant_count}，Redis计数：{lock.get_redis_count()}")  # 1,1

    # 第二次重入
    lock.acquire()
    print(f"第二次重入 - 本地计数：{lock.reentrant_count}，Redis计数：{lock.get_redis_count()}")  # 2,2

    # 第三次重入
    lock.acquire()
    print(f"第三次重入 - 本地计数：{lock.reentrant_count}，Redis计数：{lock.get_redis_count()}")  # 3,3

    # 第一次解锁
    lock.release()
    print(f"第一次解锁 - 本地计数：{lock.reentrant_count}，Redis计数：{lock.get_redis_count()}")  # 2,2

    # 第二次解锁
    lock.release()
    print(f"第二次解锁 - 本地计数：{lock.reentrant_count}，Redis计数：{lock.get_redis_count()}")  # 1,1

    # 第三次解锁（删除锁）
    lock.release()
    print(f"第三次解锁 - 本地计数：{lock.reentrant_count}，Redis计数：{lock.get_redis_count()}")  # 0,0