import threading
import time

# 创建事件对象
event = threading.Event()

def waiter():
    print("等待者等待事件发生...")
    event.wait()  # 阻塞直到事件被设置
    print("等待者检测到事件，继续执行")

def setter():
    time.sleep(3)  # 模拟工作
    print("设置者设置事件")
    event.set()  # 设置事件

t1 = threading.Thread(target=waiter)
t2 = threading.Thread(target=setter)

t1.start()
t2.start()

t1.join()
t2.join()

print("事件示例完成")