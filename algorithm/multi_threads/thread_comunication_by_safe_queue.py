import threading
from queue import Queue, LifoQueue, PriorityQueue
import time

# 1. 普通队列 (FIFO)
fifo_queue = Queue()

def fifo_worker(id):
    fifo_queue.put(f"任务-{id}")
    time.sleep(0.1)
    if not fifo_queue.empty():
        item = fifo_queue.get()
        print(f"FIFO工人 {id} 处理了 {item}")

# 2. 栈 (LIFO)
lifo_queue = LifoQueue()

def lifo_worker(id):
    lifo_queue.put(f"任务-{id}")
    time.sleep(0.1)
    if not lifo_queue.empty():
        item = lifo_queue.get()
        print(f"LIFO工人 {id} 处理了 {item}")

# 3. 优先队列
priority_queue = PriorityQueue()

def priority_worker(id, priority):
    priority_queue.put((priority, f"任务-{id}"))
    time.sleep(0.1)
    if not priority_queue.empty():
        priority, item = priority_queue.get()
        print(f"Priority工人 {id} (优先级:{priority}) 处理了 {item}")

# 测试FIFO
print("FIFO队列测试:")
threads = []
for i in range(5):
    t = threading.Thread(target=fifo_worker, args=(i,))
    threads.append(t)
    t.start()
for t in threads:
    t.join()

# 测试LIFO
print("\nLIFO队列测试:")
threads = []
for i in range(5):
    t = threading.Thread(target=lifo_worker, args=(i,))
    threads.append(t)
    t.start()
for t in threads:
    t.join()

# 测试Priority
print("\n优先队列测试:")
threads = []
for i, p in enumerate([3, 1, 4, 2, 0]):
    t = threading.Thread(target=priority_worker, args=(i, p))
    threads.append(t)
    t.start()
for t in threads:
    t.join()

print("\n所有线程安全数据结构测试完成")