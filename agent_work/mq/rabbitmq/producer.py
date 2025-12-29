# producer.py
import json
from config import RABBITMQ_CONFIG
import pika


class OrderProducer:
    def __init__(self):
        # 创建连接凭证
        credentials = pika.PlainCredentials(
            RABBITMQ_CONFIG["username"],
            RABBITMQ_CONFIG["password"]
        )
        # 建立连接
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=RABBITMQ_CONFIG["host"],
                port=RABBITMQ_CONFIG["port"],
                credentials=credentials
            )
        )
        self.channel = self.connection.channel()

        # 声明交换机和队列（幂等操作，多次声明不影响）
        self.channel.exchange_declare(
            exchange=RABBITMQ_CONFIG["exchange_name"],
            exchange_type="direct",  # 直连交换机，精准路由
            durable=True  # 持久化，避免重启丢失
        )
        self.channel.queue_declare(
            queue=RABBITMQ_CONFIG["queue_name"],
            durable=True  # 队列持久化
        )
        # 绑定交换机和队列
        self.channel.queue_bind(
            exchange=RABBITMQ_CONFIG["exchange_name"],
            queue=RABBITMQ_CONFIG["queue_name"],
            routing_key=RABBITMQ_CONFIG["routing_key"]
        )

    def send_order_message(self, order_data):
        """
        发送订单消息
        :param order_data: 订单字典，示例：{"order_id": "O20250520001", "user_id": "U1001", "amount": 99.9}
        """
        # 消息持久化（delivery_mode=2）
        properties = pika.BasicProperties(delivery_mode=2)
        # 发送消息
        self.channel.basic_publish(
            exchange=RABBITMQ_CONFIG["exchange_name"],
            routing_key=RABBITMQ_CONFIG["routing_key"],
            body=json.dumps(order_data, ensure_ascii=False),
            properties=properties
        )
        print(f"[生产者] 订单消息发送成功：{order_data['order_id']}")

    def close(self):
        self.connection.close()


# 测试生产者
if __name__ == "__main__":
    producer = OrderProducer()
    # 模拟下单场景，发送3条订单消息
    for i in range(3):
        order_id = f"O20250520{str(i).zfill(3)}"
        order_data = {
            "order_id": order_id,
            "user_id": "U1001",
            "amount": 99.9 + i,
            "create_time": "2025-05-20 12:00:00"
        }
        producer.send_order_message(order_data)
    producer.close()