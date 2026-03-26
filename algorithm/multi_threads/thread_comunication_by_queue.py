import threading
import queue
import time
import random

def producer(q, id):
    for i in range(5):
        item = f"产品-{id}-{i}"
        q.put(item)
        print(f"生产者 {id} 生产了 {item}")
        time.sleep(random.random())

def consumer(q, id):
    while True:
        item = q.get()
        if item is None:  # 结束信号
            q.put(None)  # 传递给其他消费者
            break
        print(f"消费者 {id} 消费了 {item}")
        time.sleep(random.random())
        q.task_done()

q = queue.Queue()
producers = []
consumers = []

# 创建2个生产者
for i in range(2):
    p = threading.Thread(target=producer, args=(q, i))
    producers.append(p)
    p.start()

# 创建3个消费者
for i in range(3):
    c = threading.Thread(target=consumer, args=(q, i))
    consumers.append(c)
    c.start()

# 等待所有生产者完成
for p in producers:
    p.join()

# 发送结束信号
q.put(None)

# 等待所有消费者完成
for c in consumers:
    c.join()

print("所有任务完成")