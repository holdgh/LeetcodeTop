from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import torch


def decorator_for_time(func):
    def wrapper(*args, **kwargs):
        # 1. 开始时间
        start_time = time.time()
        result = func(*args, **kwargs)  # 调用原函数
        # 2. 截止时间
        end_time = time.time()
        print(f"操作{func.__name__}耗时：{end_time - start_time}秒")
        return result

    return wrapper


def triple(x: int):
    """
     GIL 的工作原理
        1、单线程执行：
        - 每个 Python 进程内，GIL 一次仅允许一个线程持有锁并执行字节码，其他线程必须等待锁释放。
        2、锁的释放时机：
        - I/O 操作（如文件读写、网络请求）时自动释放。
        - 固定间隔（如 CPython 每执行 100 个字节码指令或 5 毫秒）强制切换线程。
    """
    torch.matmul(torch.randn(x + 1, x + 1), torch.randn(x + 1, x + 1))  # 进行矩阵乘法运算，增加cpu密集任务的复杂度，使得多进程发挥功效
    return x * x * x


@decorator_for_time
def triple_by_executor(inputs: list[int]):
    with ProcessPoolExecutor(
            max_workers=4) as executor:  # 多进程可以跳过python的全局解释器锁限制，以去除相应时间开销。对于微秒级任务，即使采用多进程，依然不如单线程方式快捷。
        results = executor.map(triple, inputs)  # map方式批量任务提交，适用于任务数量确定的场景，结果有序，代码简单
        return list(results)


@decorator_for_time
def triple_by_executor_submit(inputs: list[int]):
    results = []
    with ProcessPoolExecutor(
            max_workers=4) as executor:  # 多进程可以跳过python的全局解释器锁限制，以去除相应时间开销。对于微秒级任务，即使采用多进程，依然不如单线程方式快捷。
        futures = {executor.submit(triple, item): item for item in inputs}  # submit方式单个任务提交，适用于任务生成器场景，动态灵活
        for future in as_completed(futures):
            x = futures[future]
            try:
                results.append(future.result())
                # print(f"triple({x}) => {future.result()}")
            except Exception as e:
                print(f"triple({x}) failed: {str(e)}")
        return results


@decorator_for_time
def triple_by_no_executor(inputs: list[int]):
    results = []
    for item in inputs:
        results.append(triple(item))
    return results


if __name__ == '__main__':
    inputs = range(20)
    # triple_by_no_executor(inputs)  # 对于单线程而言，耗时会随着任务量的增加而以更快速的方式增加【非线性增长】，比如500个任务耗时6秒，那么2000个人则耗时达到90多秒。
    triple_by_executor(inputs)
    triple_by_executor_submit(inputs)
