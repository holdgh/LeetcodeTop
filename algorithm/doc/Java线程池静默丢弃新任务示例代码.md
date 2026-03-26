以下是 Java 线程池静默丢弃新任务的完整示例代码，使用 `DiscardPolicy` 实现任务被拒绝时的无提示丢弃：

### **1. 基础示例（静默丢弃）**
```java
import java.util.concurrent.*;

public class DiscardPolicyDemo {
    public static void main(String[] args) {
        // 创建线程池（核心线程2，最大线程4，队列容量2，拒绝策略为DiscardPolicy）
        ThreadPoolExecutor executor = new ThreadPoolExecutor(
            2,  // 核心线程数
            4,  // 最大线程数
            60, TimeUnit.SECONDS,
            new ArrayBlockingQueue<>(2),  // 任务队列容量
            new ThreadPoolExecutor.DiscardPolicy()  // 静默丢弃新任务
        );

        // 提交6个任务（2核心线程 + 2队列 + 2临时线程 → 第7个任务将被静默丢弃）
        for (int i = 1; i <= 7; i++) {
            final int taskId = i;
            executor.execute(() -> {
                try {
                    Thread.sleep(1000); // 模拟任务执行
                    System.out.println("任务 " + taskId + " 由线程 " + 
                                      Thread.currentThread().getName() + " 执行");
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            });
            System.out.println("提交任务 " + taskId);
        }

        executor.shutdown();
    }
}
```

#### **输出说明**
- 任务1、2由核心线程直接执行。
- 任务3、4进入队列等待。
- 任务5、6触发线程池扩容到最大线程数（4线程）执行。
- **任务7**被静默丢弃（无任何日志或异常）。

---

### **2. 增强版（记录丢弃的任务）**
如果需要记录被丢弃的任务（但不中断程序），可自定义拒绝策略：
```java
import java.util.concurrent.*;

public class EnhancedDiscardPolicyDemo {
    public static void main(String[] args) {
        // 自定义拒绝策略：记录被丢弃的任务
        RejectedExecutionHandler loggingDiscardPolicy = (runnable, executor) -> {
            System.out.println("警告：任务 " + runnable + " 被静默丢弃");
            // 可扩展：将任务存入数据库或重试队列
        };

        ThreadPoolExecutor executor = new ThreadPoolExecutor(
            2, 
            4,
            60, TimeUnit.SECONDS,
            new ArrayBlockingQueue<>(2),
            loggingDiscardPolicy  // 使用自定义策略
        );

        for (int i = 1; i <= 7; i++) {
            final int taskId = i;
            executor.execute(() -> {
                try {
                    Thread.sleep(1000);
                    System.out.println("任务 " + taskId + " 执行成功");
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            });
        }

        executor.shutdown();
    }
}
```

#### **输出示例**
```
提交任务 1
提交任务 2
提交任务 3
提交任务 4
提交任务 5
提交任务 6
警告：任务 java.util.concurrent.ThreadPoolExecutor$Worker@1b6d3586 被静默丢弃
任务 1 执行成功
任务 2 执行成功
任务 5 执行成功
任务 6 执行成功
任务 3 执行成功
任务 4 执行成功
```

---

### **3. 关键参数说明**
| **参数**               | **值**           | **作用**                                                                 |
|------------------------|------------------|-------------------------------------------------------------------------|
| `corePoolSize`         | 2                | 核心线程数（长期存活的线程）。                                           |
| `maximumPoolSize`      | 4                | 线程池最大容量（含核心线程）。                                           |
| `workQueue`           | `ArrayBlockingQueue(2)` | 任务队列容量，超过后触发扩容或拒绝策略。                                 |
| `handler`             | `DiscardPolicy`  | 拒绝策略：静默丢弃新任务。                                               |

---

### **4. 应用场景**
- **监控日志上报**：丢弃非关键日志避免内存溢出。
- **实时数据流处理**：丢弃旧数据确保处理最新消息。
- **高吞吐量服务**：短暂过载时丢弃部分请求保护系统。

---

### **5. 注意事项**
1. **资源监控**：静默丢弃可能导致任务丢失，需配合监控告警（如队列堆积告警）。
2. **队列选择**：`ArrayBlockingQueue` 有界队列才能触发拒绝策略，`LinkedBlockingQueue` 无界队列会一直堆积。
3. **线程池关闭**：务必调用 `shutdown()` 避免资源泄漏。