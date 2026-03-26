# 服务端，作用生产者与消费者之间的通信中介，起到了一个消息队列的作用，但是仅支持同一服务器
from multiprocessing import Queue
from multiprocessing.managers import BaseManager


class QueueManager(BaseManager): pass


if __name__ == '__main__':
    queue1 = Queue()
    QueueManager.register('producer_1', callable=lambda: queue1)
    queue2 = Queue()
    QueueManager.register('producer_2', callable=lambda: queue2)
    m = QueueManager(address=('', 50000), authkey=b'abc')
    s = m.get_server()
    s.serve_forever()
