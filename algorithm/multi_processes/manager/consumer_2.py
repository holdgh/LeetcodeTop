# 消费者，相对于服务端的一种请求数据的客户端
from multiprocessing.managers import BaseManager


class QueueManager(BaseManager): pass


if __name__ == '__main__':
    QueueManager.register('producer_2')
    m = QueueManager(address=('localhost', 50000), authkey=b'abc')
    m.connect()
    queue = m.producer_2()

    while True:
        msg = queue.get()
        if msg == "END":
            break
        print(f"消费者接收: {msg}")
