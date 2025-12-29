# consumer.py
import json
import threading
from config import RABBITMQ_CONFIG, order_id_set
import pika


class OrderConsumer:
    def __init__(self, thread_count=3):
        """
        初始化消费者
        :param thread_count: 并发消费线程数
        """
        self.thread_count = thread_count
        self.credentials = pika.PlainCredentials(
            RABBITMQ_CONFIG["username"],
            RABBITMQ_CONFIG["password"]
        )

    def process_order(self, channel, method, properties, body):
        """
        订单处理核心逻辑（含幂等性校验）
        """
        try:
            order_data = json.loads(body.decode("utf-8"))
            order_id = order_data["order_id"]

            # 1. 幂等性校验：已处理的订单直接确认
            if order_id in order_id_set:
                print(f"[消费者] 订单 {order_id} 已处理，跳过消费")
                channel.basic_ack(delivery_tag=method.delivery_tag)
                return

            # 2. 模拟业务处理：库存扣减、物流通知等
            print(f"[消费者] 开始处理订单：{order_id}，金额：{order_data['amount']}")
            # TODO: 实际业务逻辑（如调用库存服务）

            # 3. 标记订单为已处理
            order_id_set.add(order_id)

            # 4. 手动 ACK 确认：消息处理成功，RabbitMQ 删除消息
            channel.basic_ack(delivery_tag=method.delivery_tag)
            print(f"[消费者] 订单 {order_id} 处理完成，已发送 ACK")

        except Exception as e:
            print(f"[消费者] 订单处理失败：{str(e)}")
            # 消费失败：拒绝消息并重新入队（或根据需求丢弃/转入死信队列）
            channel.basic_nack(
                delivery_tag=method.delivery_tag,
                requeue=True,  # True=重新入队，False=丢弃
                multiple=False
            )

    def consume(self):
        """单线程消费"""
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=RABBITMQ_CONFIG["host"],
                port=RABBITMQ_CONFIG["port"],
                credentials=self.credentials,
                heartbeat=600  # 心跳超时时间，避免连接断开
            )
        )
        channel = connection.channel()

        # 声明队列（与生产者保持一致）
        channel.queue_declare(
            queue=RABBITMQ_CONFIG["queue_name"],
            durable=True
        )

        # 关键配置：公平分发，避免单个消费者堆积过多消息
        channel.basic_qos(prefetch_count=1)

        # 注册消费回调函数，关闭自动 ACK（auto_ack=False）
        channel.basic_consume(
            queue=RABBITMQ_CONFIG["queue_name"],
            on_message_callback=self.process_order,
            auto_ack=False  # 核心：手动 ACK 模式
        )

        print(f"[消费者] 单线程消费启动，等待订单消息...")
        channel.start_consuming()

    def start_concurrent_consume(self):
        """多线程并发消费"""
        threads = []
        for i in range(self.thread_count):
            t = threading.Thread(target=self.consume, name=f"Consumer-Thread-{i + 1}")
            t.start()
            threads.append(t)

        for t in threads:
            t.join()


# 测试消费者
if __name__ == "__main__":
    consumer = OrderConsumer(thread_count=3)  # 3线程并发消费
    consumer.start_concurrent_consume()