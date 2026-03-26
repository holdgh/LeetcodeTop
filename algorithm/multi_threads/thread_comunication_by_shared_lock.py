import threading

# 共享变量
counter = 0
lock = threading.Lock()

def worker():
    global counter
    for _ in range(100000):
        with lock:  # 使用锁保证线程安全
            counter += 1

threads = []
for _ in range(5):
    t = threading.Thread(target=worker)
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print("Final counter value:", counter)  # 应该输出500000