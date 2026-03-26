以下是 Java 线程池使用 **`DiscardOldestPolicy`** 策略的完整示例代码，该策略会丢弃队列中最旧的任务，然后尝试重新提交新任务：

---

### **1. 基础示例（丢弃最老任务）**
```java
import java.util.concurrent.*;

public class DiscardOldestPolicyDemo {
    public static void main(String[] args) {
        // 创建线程池（核心线程2，最大线程4，队列容量2，拒绝策略为DiscardOldestPolicy）
        ThreadPoolExecutor executor = new ThreadPoolExecutor(
            2,  // 核心线程数
            4,  // 最大线程数
            60, TimeUnit.SECONDS,
            new ArrayBlockingQueue<>(2),  // 任务队列容量
            new ThreadPoolExecutor.DiscardOldestPolicy()  // 丢弃队列头部最旧任务
        );

        // 提交6个任务（2核心线程 + 2队列 + 2临时线程 → 第7个任务触发丢弃最旧任务）
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
- **任务1、2**：由核心线程直接执行。
- **任务3、4**：进入队列等待。
- **任务5、6**：触发线程池扩容到最大线程数（4线程）执行。
- **任务7**：提交时队列已满，丢弃队列头部最旧的 **任务3**，并将任务7加入队列末尾。
- **最终执行的任务**：1, 2, 5, 6, 4, 7（任务3被丢弃）。

---

### **2. 增强版（记录被丢弃的任务）**
若需记录被丢弃的任务信息，可自定义拒绝策略：
```java
import java.util.concurrent.*;

public class EnhancedDiscardOldestPolicyDemo {
    public static void main(String[] args) {
        // 自定义拒绝策略：记录被丢弃的旧任务
        RejectedExecutionHandler loggingDiscardOldestPolicy = (runnable, executor) -> {
            if (executor.getQueue() instanceof BlockingQueue) {
                BlockingQueue<Runnable> queue = (BlockingQueue<Runnable>) executor.getQueue();
                Runnable oldestTask = queue.poll(); // 移除并获取队列头部任务
                System.out.println("警告：丢弃旧任务 " + oldestTask);
                executor.execute(runnable); // 重试提交新任务
            }
        };

        ThreadPoolExecutor executor = new ThreadPoolExecutor(
            2, 
            4,
            60, TimeUnit.SECONDS,
            new ArrayBlockingQueue<>(2),
            loggingDiscardOldestPolicy  // 使用自定义策略
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
警告：丢弃旧任务 java.util.concurrent.FutureTask@1b6d3586
提交任务 7
任务 1 执行成功
任务 2 执行成功
任务 5 执行成功
任务 6 执行成功
任务 4 执行成功
任务 7 执行成功
```
（任务3被丢弃，任务7加入队列）

---

### **3. 关键参数说明**
| **参数**               | **值**                 | **作用**                                                                 |
|------------------------|------------------------|-------------------------------------------------------------------------|
| `corePoolSize`         | 2                      | 核心线程数（长期存活）。                                                 |
| `maximumPoolSize`      | 4                      | 线程池最大容量（含核心线程）。                                           |
| `workQueue`           | `ArrayBlockingQueue(2)` | 有界队列，容量为2。                                                     |
| `handler`             | `DiscardOldestPolicy`   | 拒绝策略：丢弃队列头部最旧任务，重试提交新任务。                         |

---

### **4. 应用场景**
- **实时数据处理**：优先处理最新数据（如股票行情，旧数据可丢弃）。
- **高优先级任务**：确保新任务能及时执行，牺牲部分旧任务。
- **流量控制**：防止队列堆积导致内存溢出。

---

### **5. 注意事项**
1. **任务丢失风险**：被丢弃的任务不会被执行，需确保业务允许。
2. **队列类型**：必须使用有界队列（如 `ArrayBlockingQueue`），无界队列（如 `LinkedBlockingQueue`）不会触发拒绝策略。
3. **监控建议**：记录被丢弃的任务数量和类型，便于问题排查。

---

### **总结**
通过 `DiscardOldestPolicy`，线程池可以在过载时优先保证新任务的执行，适合对实时性要求高的场景。若需更精细控制（如记录日志或降级处理），可自定义拒绝策略。