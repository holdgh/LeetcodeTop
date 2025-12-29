import time
import logging
from functools import wraps
from typing import Callable, Any

# 日志配置（同上）
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# 异步函数计时装饰器（适配 async def 函数）
def async_timer_with_mark(mark_param_name: str = ""):
    def async_timer(func: Callable) -> Callable:
        @wraps(func)  # 保留原函数的名称、文档字符串等属性
        async def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            # 提取 标识参数名
            mark_param_value = kwargs.get(mark_param_name, "")
            # 记录函数名和参数（参数过长时截取）
            # args_str = ", ".join([str(arg)[:30] for arg in args])  # 参数截取30字
            # kwargs_str = ", ".join([f"{k}={v[:30]}" for k, v in kwargs.items()])
            # logger.info(f"【{func.__name__}】函数启动，参数：{args_str} | {kwargs_str}")

            try:
                # 执行原函数
                result = await func(*args, **kwargs)
                # 计算耗时
                elapsed_time = (time.perf_counter() - start_time) * 1000
                logger.info(f"【{func.__name__}】对{mark_param_value}消息的操作完成，耗时：{elapsed_time:.2f}ms")
                return result
            except Exception as e:
                elapsed_time = (time.perf_counter() - start_time) * 1000
                logger.error(
                    f"【{func.__name__}对{mark_param_value}消息的操作失败，耗时：{elapsed_time:.2f}ms，错误：{str(e)}",
                    exc_info=True  # 打印异常堆栈，便于排查
                )
                raise

        return wrapper
    return async_timer


# 同步函数计时装饰器（如果有同步函数需要计时）
def sync_timer_with_mark(mark_param_name: str = ""):
    def sync_timer(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()# 提取 标识参数名
            mark_param_value = kwargs.get(mark_param_name, "")
            # args_str = ", ".join([str(arg)[:30] for arg in args])
            # kwargs_str = ", ".join([f"{k}={v[:30]}" for k, v in kwargs.items()])
            # logger.info(f"【{func.__name__}】函数启动，参数：{args_str} | {kwargs_str}")

            try:
                result = func(*args, **kwargs)
                elapsed_time = (time.perf_counter() - start_time) * 1000
                logger.info(f"【{func.__name__}】对{mark_param_value}消息的操作完成，耗时：{elapsed_time:.2f}ms")
                return result
            except Exception as e:
                elapsed_time = (time.perf_counter() - start_time) * 1000
                logger.error(
                    f"【{func.__name__}对{mark_param_value}消息的操作失败，耗时：{elapsed_time:.2f}ms，错误：{str(e)}",
                    exc_info=True  # 打印异常堆栈，便于排查
                )
                raise

        return wrapper
    return sync_timer