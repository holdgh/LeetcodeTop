在 Python 中，`yield` 是用于定义**生成器函数（generator function）**的关键字，它的核心作用是**实现惰性计算（lazy evaluation）**，允许函数暂停执行并保留当前状态，后续再恢复执行。这使得生成器可以高效处理大数据、无限序列或流式数据。

---

### 主要作用：
1. **创建生成器**  
   当一个函数包含 `yield` 时，调用该函数会返回一个**生成器对象**，而不会立即执行函数体。
   ```python
   def my_generator():
       yield 1
       yield 2
       yield 3

   gen = my_generator()  # 返回生成器对象，代码不会立即执行
   print(next(gen))  # 输出 1
   print(next(gen))  # 输出 2
   ```

2. **暂停与恢复执行**  
   每次执行到 `yield` 时，函数会**暂停**并返回 `yield` 后的值；下次通过 `next()` 或迭代时，从暂停处继续执行。
   ```python
   def count_up():
       n = 0
       while True:
           yield n
           n += 1

   counter = count_up()
   print(next(counter))  # 0
   print(next(counter))  # 1 (从 yield 处恢复，继续执行 n += 1)
   ```

3. **节省内存**  
   生成器**按需生成值**，避免一次性加载所有数据到内存，适合处理大文件或无限序列。
   ```python
   def read_large_file(file_path):
       with open(file_path) as f:
           for line in f:
               yield line.strip()  # 逐行生成，不一次性读取全部内容

   for line in read_large_file("data.txt"):
       process(line)  # 逐行处理
   ```

4. **双向通信**  
   通过 `send()` 方法，可以向生成器内部发送数据（协程的基础）。
   ```python
   def coroutine():
       while True:
           x = yield  # 接收外部发送的值
           print(f"Received: {x}")

   c = coroutine()
   next(c)        # 启动生成器（执行到 yield 处暂停）
   c.send(10)     # 输出 "Received: 10"
   ```

---

### 与 `return` 的区别：
| 特性              | `yield`                          | `return`                 |
|-------------------|----------------------------------|--------------------------|
| **函数类型**       | 生成器函数                       | 普通函数                 |
| **执行后状态**     | 暂停，保留局部变量               | 终止，释放局部变量       |
| **返回值**         | 生成值，可通过迭代或 `next()` 获取 | 直接返回结果             |
| **内存效率**       | 高（按需生成）                   | 低（一次性返回所有结果） |

---

### 典型应用场景：
- **处理大型数据集**（如逐行读取文件）
- **生成无限序列**（如斐波那契数列、素数序列）
- **实现协程和异步编程**
- **管道式数据处理**（多个生成器串联）

示例：生成斐波那契数列
```python
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

fib = fibonacci()
print([next(fib) for _ in range(10)])  # 输出前10项: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

通过 `yield`，Python 能以简洁的语法实现高效的迭代逻辑，是处理流式数据的核心工具之一。