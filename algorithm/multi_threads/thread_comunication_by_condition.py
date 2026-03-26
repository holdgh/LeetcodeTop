import threading
import time

shared_list = []
condition = threading.Condition()

def consumer():
    global shared_list
    with condition:
        print("消费者等待条件满足...")
        condition.wait_for(lambda: len(shared_list) > 0)  # 等待条件满足
        item = shared_list.pop()
        print(f"消费者消费了: {item}")

def producer():
    global shared_list
    time.sleep(2)  # 模拟工作
    with condition:
        item = "新产品"
        shared_list.append(item)
        print(f"生产者生产了: {item}")
        condition.notify()  # 通知等待的线程

t1 = threading.Thread(target=consumer)
t2 = threading.Thread(target=producer)

t1.start()
t2.start()

t1.join()
t2.join()

print("条件变量示例完成")