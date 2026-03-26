import threading
import time

# 创建屏障，等待4个线程到达
barrier = threading.Barrier(4)

def worker(id):
    print(f"工人 {id} 开始工作")
    time.sleep(id)  # 模拟不同长度的工作
    print(f"工人 {id} 完成工作，等待其他工人")
    barrier.wait()  # 等待所有线程到达
    print(f"工人 {id} 继续后续工作")

threads = []
for i in range(4):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print("屏障示例完成")