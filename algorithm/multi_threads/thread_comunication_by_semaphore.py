import threading
import time

# 限制同时只有3个线程可以访问资源
semaphore = threading.Semaphore(3)

def worker(id):
    print(f"工人 {id} 等待获取信号量...")
    with semaphore:
        print(f"工人 {id} 获得了信号量")
        time.sleep(2)  # 模拟工作
    print(f"工人 {id} 释放了信号量")

threads = []
for i in range(10):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print("信号量示例完成")