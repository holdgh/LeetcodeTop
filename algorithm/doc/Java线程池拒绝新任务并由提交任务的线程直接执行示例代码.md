以下是 Java 线程池使用 **`CallerRunsPolicy`** 的完整示例代码，该策略会在线程池和队列满时，由提交任务的线程直接执行该任务（而非丢弃或抛出异常）：

---

### **1. 基础示例（由调用线程直接执行）**
```java
import java.util.concurrent.*;

public class CallerRunsPolicyDemo {
    public static void main(String[] args) {
        // 创建线程池（核心线程2，最大线程4，队列容量2，拒绝策略为CallerRunsPolicy）
        ThreadPoolExecutor executor = new ThreadPoolExecutor(
            2,  // 核心线程数
            4,  // 最大线程数
            60, TimeUnit.SECONDS,
            new ArrayBlockingQueue<>(2),  // 任务队列容量
            new ThreadPoolExecutor.CallerRunsPolicy()  // 由提交线程直接执行
        );

        // 提交6个任务（2核心线程 + 2队列 + 2临时线程 → 第7个任务由主线程直接执行）
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
- **任务1、2**：由核心线程（`pool-1-thread-1` 和 `pool-1-thread-2`）执行。
- **任务3、4**：进入队列等待。
- **任务5、6**：触发线程池扩容到最大线程数（`pool-1-thread-3` 和 `pool-1-thread-4`）执行。
- **任务7**：线程池和队列已满，由提交任务的线程（`main` 主线程）直接执行。
- **输出示例**：
  ```
  提交任务 1
  提交任务 2
  提交任务 3
  提交任务 4
  提交任务 5
  提交任务 6
  提交任务 7
  任务 1 由线程 pool-1-thread-1 执行
  任务 7 由线程 main 执行  // 主线程直接执行！
  任务 2 由线程 pool-1-thread-2 执行
  任务 5 由线程 pool-1-thread-3 执行
  任务 6 由线程 pool-1-thread-4 执行
  任务 3 由线程 pool-1-thread-1 执行
  任务 4 由线程 pool-1-thread-2 执行
  ```

---

### **2. 增强版（记录任务来源）**
若需区分任务是由线程池还是调用线程执行的，可添加日志：
```java
import java.util.concurrent.*;

public class EnhancedCallerRunsPolicyDemo {
    public static void main(String[] args) {
        ThreadPoolExecutor executor = new ThreadPoolExecutor(
            2, 
            4,
            60, TimeUnit.SECONDS,
            new ArrayBlockingQueue<>(2),
            new ThreadPoolExecutor.CallerRunsPolicy()
        );

        for (int i = 1; i <= 7; i++) {
            final int taskId = i;
            executor.execute(() -> {
                String executorType = Thread.currentThread().getName().startsWith("pool-") ? 
                                     "线程池线程" : "调用线程";
                System.out.println("任务 " + taskId + " 由 " + executorType + " 执行");
            });
        }

        executor.shutdown();
    }
}
```

#### **输出示例**
```
任务 1 由 线程池线程 执行
任务 2 由 线程池线程 执行
任务 5 由 线程池线程 执行
任务 6 由 线程池线程 执行
任务 7 由 调用线程 执行  // 主线程直接执行！
任务 3 由 线程池线程 执行
任务 4 由 线程池线程 执行
```

---

### **3. 关键参数说明**
| **参数**               | **值**                 | **作用**                                                                 |
|------------------------|------------------------|-------------------------------------------------------------------------|
| `corePoolSize`         | 2                      | 核心线程数（长期存活）。                                                 |
| `maximumPoolSize`      | 4                      | 线程池最大容量（含核心线程）。                                           |
| `workQueue`           | `ArrayBlockingQueue(2)` | 有界队列，容量为2。                                                     |
| `handler`             | `CallerRunsPolicy`      | 拒绝策略：由提交任务的线程直接执行。                                     |

---

### **4. 应用场景**
- **流量削峰**：当线程池过载时，通过调用线程直接执行任务，自然限流。
- **任务保序**：避免任务丢失（如订单处理，必须保证所有任务完成）。
- **优雅降级**：在高负载时，由调用线程执行任务，确保系统不崩溃。

---

### **5. 注意事项**
1. **调用线程阻塞**：若任务耗时较长，提交任务的线程（如主线程）会被阻塞。
2. **队列类型**：必须使用有界队列（如 `ArrayBlockingQueue`），无界队列不会触发拒绝策略。
3. **性能影响**：调用线程直接执行任务可能影响原有逻辑（如 Web 服务的请求线程被占用）。

---

### **总结**
`CallerRunsPolicy` 是一种温和的拒绝策略，适合需要 **保证任务绝对执行** 且 **允许短暂阻塞调用线程** 的场景。相比直接丢弃任务或抛出异常，它能更优雅地应对瞬时高负载。