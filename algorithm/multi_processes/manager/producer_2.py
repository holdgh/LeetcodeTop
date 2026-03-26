# 生产者，相对于服务端的一种产生数据的客户端
from multiprocessing.managers import BaseManager


class QueueManager(BaseManager): pass


if __name__ == '__main__':
    QueueManager.register('producer_2')
    m = QueueManager(address=('localhost', 50000), authkey=b'abc')
    m.connect()
    queue = m.producer_2()  # 注意此处与register方法注册的名称一致

    for i in range(5):
        queue.put(f"producer_2的消息 {i}")
        print(f"生产者发送: producer_2的消息 {i}")
    queue.put('END')
