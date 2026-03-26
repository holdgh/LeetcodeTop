在 Python 的 `concurrent.futures.ThreadPoolExecutor` 中，确实没有显式的 **核心线程数（corePoolSize）** 和 **最大线程数（maximumPoolSize）** 的分离参数（类似 Java 的 `ThreadPoolExecutor`）。这是 Python 线程池设计上的简化选择，核心逻辑如下：

---

### **1. Python 线程池的简化设计**
Python 的 `ThreadPoolExecutor` 仅通过 **`max_workers`** 参数控制线程池的最大线程数。其设计目标是 **轻量易用**，而非提供复杂的线程池调控策略。以下是关键点：

| **设计特点**                | **说明**                                                                 |
|----------------------------|-------------------------------------------------------------------------|
| **统一线程数**              | 所有线程均为“核心线程”，没有“核心”与“非核心”的区分。                      |
| **固定线程池大小**          | 线程池初始化时直接创建 `max_workers` 个线程，任务队列满后不会动态扩容。    |
| **简单任务调度**            | 任务直接提交到队列，线程按需从队列中取任务执行。                          |

---

### **2. 与 Java 线程池的对比**
| **特性**                   | **Python `ThreadPoolExecutor`**          | **Java `ThreadPoolExecutor`**          |
|----------------------------|------------------------------------------|----------------------------------------|
| **线程类型区分**           | 无（所有线程均为核心线程）               | 核心线程和非核心线程                   |
| **动态扩容**               | 不支持（线程数固定为 `max_workers`）     | 支持（根据队列和 `maximumPoolSize` 扩容） |
| **任务队列行为**           | 队列满后直接阻塞提交线程                 | 队列满后触发扩容或拒绝策略             |

---

### **3. Python 线程池的工作流程**
Python 的线程池行为更接近 Java 中 **固定大小线程池（`Executors.newFixedThreadPool`）** 的逻辑：
1. **初始化线程池**：创建 `max_workers` 个线程。
2. **任务提交**：
   - 如果存在空闲线程，立即执行任务。
   - 否则，任务进入队列等待。
3. **队列管理**：队列默认无界（理论上可能引发内存问题），但实践中需用户自行控制提交速率。

**示例代码**：
```python
from concurrent.futures import ThreadPoolExecutor

def task(n):
    print(f"任务 {n} 由线程执行")

# 创建固定大小为 3 的线程池
with ThreadPoolExecutor(max_workers=3) as executor:
    for i in range(10):
        executor.submit(task, i)
```

---

### **4. 为何不提供核心线程数设置？**
Python 的设计选择主要基于以下原因：
1. **简化使用**：  
   Python 强调“简单优于复杂”，减少参数配置门槛，适用于大多数轻量级并发场景。

2. **GIL（全局解释器锁）限制**：  
   由于 GIL 的存在，Python 线程在 CPU 密集型任务中无法真正并行，复杂的线程池调控意义有限。

3. **适用场景定位**：  
   Python 的线程池更适合 **I/O 密集型任务**（如网络请求、文件读写），此时固定线程数已足够高效。

---

### **5. 替代方案**
若需类似 Java 的动态线程池行为，可通过以下方式实现：
#### **(1) 使用 `ProcessPoolExecutor`**
绕过 GIL，适用于 CPU 密集型任务，但进程间通信开销较大：
```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=4) as executor:
    executor.map(cpu_intensive_task, data)
```

#### **(2) 第三方库扩展**
如 `billiard`（支持复杂线程池配置）或 `celery`（分布式任务队列）：
```python
from billiard.pool import Pool

pool = Pool(processes=4, initializer=init_worker)
results = pool.map(complex_task, args_list)
```

#### **(3) 手动实现动态线程池**
通过组合队列和线程模块自行封装：
```python
import threading
import queue

class DynamicThreadPool:
    def __init__(self, core_size, max_size):
        self.core_size = core_size
        self.max_size = max_size
        self.task_queue = queue.Queue()
        self.threads = []

    def start(self):
        for _ in range(self.core_size):
            self._add_thread()

    def _add_thread(self):
        thread = threading.Thread(target=self._worker)
        thread.start()
        self.threads.append(thread)

    def _worker(self):
        while True:
            task = self.task_queue.get()
            if task is None:  # 终止信号
                break
            task()
            self.task_queue.task_done()

    def submit(self, task):
        if self.task_queue.qsize() > 0 and len(self.threads) < self.max_size:
            self._add_thread()
        self.task_queue.put(task)

    def shutdown(self):
        for _ in self.threads:
            self.task_queue.put(None)
        for thread in self.threads:
            thread.join()
```

---

### **6. 总结**
Python 的 `ThreadPoolExecutor` 不区分核心线程数和最大线程数，主要出于 **简化设计** 和 **适用场景限制**（如 GIL 的存在）。对于需要动态线程管理的场景，建议：
- 使用 `max_workers` 合理配置线程数。
- 结合进程池或第三方库扩展功能。
- 在 I/O 密集型任务中，固定线程池已能发挥最佳性能。

这种设计降低了多线程编程的门槛，符合 Python“简单、明确”的哲学。