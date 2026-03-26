from multiprocessing import Process, Queue
import time


def producer(q: Queue):  # 生产者逻辑
    for i in range(5):
        # 持久化数据后，打印消息内容
        q.put(f"消息 {i}")
        print(f"生产者发送: 消息 {i}")
        time.sleep(0.5)


def consumer(q):
    while True:
        msg = q.get()
        if msg == "END":  # 在消费者逻辑中定义结束信号
            break
        print(f"消费者接收: {msg}")


if __name__ == '__main__':
    # Queue底层使用了管道和信号量实现同步
    q = Queue()  # 初始化消息队列。消息队列是主进程中创建的共享队列，然后传递给两个子进程使用。这种模式适用于同一主进程下的多个子进程间通信
    # target表示当前进程要执行的操作，args表示执行操作所需参数
    p1 = Process(target=producer, args=(q,))  # 进程1--生产者
    p2 = Process(target=consumer, args=(q,))  # 进程2--消费者

    p1.start()  # 启动进程
    p2.start()

    p1.join()  # 等待进程p1结束
    q.put("END")  # 发送结束信号
    p2.join()  # 等待进程p2结束
