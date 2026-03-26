`django.db.close_old_connections()` 是 Django 中一个用于**主动关闭闲置或过期数据库连接**的工具方法，主要目的是**防止数据库连接泄漏**或**避免因长时间闲置导致的数据库连接超时问题**。它在以下场景中尤为重要：

---

### **核心作用**
1. **关闭闲置连接**  
   数据库服务端（如 MySQL、PostgreSQL）通常会对空闲连接设置超时时间（例如 `wait_timeout`）。如果 Django 长期未使用的连接未关闭，服务端可能会主动断开连接，导致后续操作抛出异常（如 `OperationalError: MySQL server has gone away`）。  
   `close_old_connections()` 会检查并关闭所有超过 `CONN_MAX_AGE`（连接最大存活时间）的闲置连接，确保后续操作使用新连接。

2. **清理无效连接**  
   如果连接已被数据库服务端关闭（如网络问题、服务端重启），但 Django 的连接池仍保留这些无效连接，调用此方法会清理它们，避免后续操作失败。

3. **多线程/异步环境下的线程安全**  
   在异步任务（如 Celery 任务）或多线程场景中，跨线程共享数据库连接可能导致状态混乱。手动调用此方法可以确保每个线程或任务使用独立的连接。

---

### **典型使用场景**
#### 1. **长时间运行的进程（如 Celery 任务）**
   ```python
   from django.db import close_old_connections

   @shared_task
   def my_celery_task():
       try:
           # 任务逻辑开始前关闭旧连接
           close_old_connections()
           # 执行数据库操作...
           User.objects.update(last_active=now())
       finally:
           # 任务结束后再次清理连接
           close_old_connections()
   ```

#### 2. **HTTP 长轮询或 WebSocket 应用**
   ```python
   from django.db import close_old_connections
   from channels.generic.websocket import WebsocketConsumer

   class MyConsumer(WebsocketConsumer):
       def receive(self, text_data):
           close_old_connections()  # 每次处理消息前清理连接
           # 执行数据库操作...
           user = User.objects.get(id=1)
           self.send(text_data=f"Hello {user.username}")
   ```

#### 3. **命令行脚本或定时任务**
   ```python
   from django.db import close_old_connections
   import time

   while True:
       close_old_connections()  # 每次循环前清理旧连接
       process_data()
       time.sleep(60)
   ```

---

### **与 Django 默认行为的区别**
- **Web 请求环境**：  
  Django 的 `DatabaseWrapper` 会在每个请求开始和结束时自动调用 `close_old_connections()`，因此通常无需手动处理。
  
- **非 Web 环境**：  
  在 Celery 任务、CLI 脚本、异步框架（如 Channels）中，Django 不会自动管理连接，需手动调用此方法。

---

### **关键参数与配置**
- **`CONN_MAX_AGE`**（在 `settings.py` 中配置）:  
  定义数据库连接的最大存活时间（秒）。默认值为 `0`（请求结束后立即关闭连接）。  
  若设置为正数（如 `300`），连接会在存活超过该时间后被关闭。`close_old_connections()` 依赖此值判断是否关闭连接。

  ```python
  # settings.py
  DATABASES = {
      'default': {
          'ENGINE': 'django.db.backends.mysql',
          'CONN_MAX_AGE': 300,  # 连接最多存活 5 分钟
      }
  }
  ```

---

### **实现原理**
1. 遍历所有已存在的数据库连接（对应 `django.db.connections`）。
2. 检查每个连接的存活时间是否超过 `CONN_MAX_AGE`。
3. 关闭超时的连接或已被服务端标记为无效的连接。

---

### **注意事项**
1. **性能影响**：  
   频繁调用 `close_old_connections()` 可能导致连接频繁重建，增加延迟。应根据场景平衡调用频率（例如在任务边界调用，而非每次数据库操作前）。

2. **连接复用**：  
   若 `CONN_MAX_AGE > 0`，Django 会复用连接，减少新建连接的开销。但在多线程环境中需确保线程安全。

3. **信号替代方案**：  
   对于 Celery，可以结合信号自动清理连接：
   ```python
   from celery.signals import task_prerun, task_postrun
   from django.db import close_old_connections

   @task_prerun.connect
   @task_postrun.connect
   def close_db_connections(**kwargs):
       close_old_connections()
   ```

---

### **示例：修复 "MySQL server has gone away" 错误**
若在长时间任务中遇到此类错误，可通过定期清理连接修复：
```python
def long_running_process():
    while True:
        close_old_connections()  # 每次循环清理旧连接
        data = fetch_data()
        process(data)
        time.sleep(10)
```

---

通过合理使用 `close_old_connections()`，可以显著提升 Django 在非 Web 环境下的数据库连接稳定性，避免因连接泄漏或超时引发的异常。