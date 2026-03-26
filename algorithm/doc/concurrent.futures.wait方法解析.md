`concurrent.futures.wait(self.future_list, timeout)` 是 Python 中用于**并发任务管理**的关键函数，其核心作用是 **等待一组异步任务（Future 对象）完成或超时**，并返回已完成和未完成的任务状态。以下是其详细解释：

---

### **功能与参数**
- **功能**：  
  阻塞当前线程，等待指定的 `Future` 对象集合完成或超时，返回一个包含 **已完成任务** 和 **未完成任务** 的元组。
  
- **参数**：
  - `self.future_list`: 由 `ThreadPoolExecutor` 或 `ProcessPoolExecutor` 生成的 `Future` 对象列表。
  - `timeout` (可选): 最大等待时间（秒）。若为 `None`，则无限等待；若超时后仍有任务未完成，直接返回当前状态。

---

### **返回值**
返回一个 `namedtuple` 对象，包含两个集合：
- **`done`**: 已完成的 `Future` 对象集合（无论成功或异常）。
- **`not_done`**: 未完成的 `Future` 对象集合（仍在运行）。

示例：
```python
from concurrent.futures import ThreadPoolExecutor, wait

def task(n):
    return n * 2

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(task, i) for i in range(5)]
    done, not_done = wait(futures, timeout=2)

print(f"已完成的任务数: {len(done)}")    # 输出结果取决于任务执行速度
print(f"未完成的任务数: {len(not_done)}")
```

---

### **典型应用场景**
1. **批量任务监控**  
   等待所有任务完成，或超时后处理部分结果：
   ```python
   done, _ = wait(future_list)
   results = [future.result() for future in done]
   ```

2. **超时控制**  
   避免因某些任务长时间阻塞主程序：
   ```python
   done, not_done = wait(future_list, timeout=10)
   if not_done:
       print("部分任务超时未完成")
   ```

3. **分批处理任务**  
   先处理已完成的任务，再处理剩余任务：
   ```python
   while futures:
       done, futures = wait(futures, timeout=5)
       process_results(done)
   ```

---

### **关键注意事项**
1. **不取消未完成任务**  
   `wait()` 仅返回状态，**不会终止未完成任务**。需手动取消：
   ```python
   for future in not_done:
       future.cancel()
   ```

2. **结合 `return_when` 细化条件**  
   通过 `return_when` 参数指定触发返回的条件（默认为 `ALL_COMPLETED`）：
   ```python
   from concurrent.futures import FIRST_COMPLETED, ALL_COMPLETED

   # 当第一个任务完成时返回
   done, not_done = wait(futures, return_when=FIRST_COMPLETED)
   ```

3. **异常处理**  
   已完成的 `Future` 可能包含异常，需调用 `result()` 时捕获：
   ```python
   for future in done:
       try:
           print(future.result())
       except Exception as e:
           print(f"任务出错: {e}")
   ```

---

### **与 `as_completed()` 的区别**
| **函数**         | **行为**                                                                 | **适用场景**                     |
|------------------|-------------------------------------------------------------------------|----------------------------------|
| `wait()`         | 批量等待任务，返回所有完成/未完成的对象                                   | 需要同时监控多个任务状态          |
| `as_completed()` | 生成一个迭代器，按任务完成顺序逐个返回 `Future` 对象                       | 实时处理每个完成的任务（流式处理） |

示例对比：
```python
# 使用 as_completed() 逐个处理
from concurrent.futures import as_completed

for future in as_completed(futures):
    print(future.result())

# 使用 wait() 批量处理
done, _ = wait(futures)
for future in done:
    print(future.result())
```

---

### **总结**
- **`wait()`** 是管理并发任务的核心工具，适用于需要 **批量等待任务完成** 或 **设置超时** 的场景。
- 结合 `timeout` 和 `return_when` 参数，可灵活控制任务执行流程。
- 需注意处理未完成任务和异常，以确保程序健壮性。