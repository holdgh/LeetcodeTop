## 详细注释：使用管道进行父子进程通信

```python
# 导入必要的模块
from multiprocessing import Process, Pipe

# 子进程函数
def child_process(conn):
    """
    子进程的执行逻辑
    :param conn: 管道连接对象，用于与父进程通信
    """
    # 子进程从管道接收父进程发送的消息
    # recv()是阻塞调用，会等待直到有消息到达
    print("子进程收到消息:", conn.recv())  # 接收消息
    
    # 子进程通过管道向父进程发送消息
    conn.send("Hello Parent!")  # 发送消息
    
    # 关闭子进程端的管道连接
    conn.close()

# 主程序入口
if __name__ == '__main__':
    """
    主进程(父进程)的执行逻辑
    """
    # 创建管道，返回两个连接对象:
    # parent_conn - 父进程使用的连接端
    # child_conn - 将传递给子进程的连接端
    # Pipe()创建的是双向管道，两端都可以发送和接收
    parent_conn, child_conn = Pipe()  # 创建管道
    
    # 创建子进程
    # target指定子进程要执行的函数
    # args传递参数，这里将child_conn传递给子进程
    p = Process(target=child_process, args=(child_conn,))
    
    # 启动子进程
    p.start()
    
    # 父进程通过管道向子进程发送消息
    # 这个消息会被子进程的conn.recv()接收
    parent_conn.send("Hello Child!")  # 发送消息
    
    # 父进程从管道接收子进程发送的消息
    # recv()是阻塞调用，会等待直到有消息到达
    print("父进程收到消息:", parent_conn.recv())  # 接收消息
    
    # 等待子进程结束
    p.join()
    
    # 父进程端的管道连接会在父进程结束时自动关闭
    # 也可以显式调用 parent_conn.close()
```

## 关键点解释

1. **父子进程识别**:
   - 父进程: `if __name__ == '__main__':` 代码块中的主程序
   - 子进程: 由 `Process()` 创建并启动的进程，执行 `child_process` 函数

2. **管道连接对应关系**:
   - `parent_conn, child_conn = Pipe()` 创建了一对相互连接的管道端点
   - 父进程保留 `parent_conn` 用于与子进程通信
   - 子进程获得 `child_conn` 用于与父进程通信
   - 这两个连接对象是相互连接的，一端发送的数据会被另一端接收

3. **通信流程**:
   ```
   父进程 parent_conn.send("Hello Child!") 
   → 
   子进程 conn.recv() 收到"Hello Child!"
   →
   子进程 conn.send("Hello Parent!") 
   →
   父进程 parent_conn.recv() 收到"Hello Parent!"
   ```

4. **重要注意事项**:
   - 管道是双向的，两端都可以发送和接收
   - 每个进程应该只使用分配给它的连接端
   - 使用完毕后应该关闭连接(特别是子进程)
   - 父进程需要调用 `join()` 等待子进程结束
   - 如果管道两端都关闭了，再尝试发送/接收会引发异常

5. **典型应用场景**:
   - 父子进程需要双向通信时
   - 需要传递Python对象(管道会自动序列化/反序列化)
   - 进程间通信量不大的情况(管道不适合传输大量数据)

这种管道通信方式是高效的进程间通信方法，特别适合需要双向通信的父子进程场景。