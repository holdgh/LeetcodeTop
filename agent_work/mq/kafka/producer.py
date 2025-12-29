# kafka_producer.py
import json
from config import KAFKA_CONFIG
from kafka import KafkaProducer

class KafkaOrderProducer:
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=KAFKA_CONFIG["bootstrap_servers"],
            value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
            acks="all",  # 等待所有副本确认，最高可靠性
            retries=3,  # 发送失败重试次数
            linger_ms=5  # 批量发送延迟，提升吞吐量
        )

    def send_order_message(self, order_data):
        """发送订单消息"""
        # 基于 order_id 路由到固定分区（保证同一订单的消息在同一分区，实现顺序消费）
        future = self.producer.send(
            topic=KAFKA_CONFIG["topic"],
            value=order_data,
            key=order_data["order_id"].encode("utf-8")  # 分区路由 Key
        )
        # 等待发送结果（同步确认）
        try:
            record_metadata = future.get(timeout=10)
            print(f"[Kafka生产者] 订单 {order_data['order_id']} 发送成功，分区：{record_metadata.partition}")
        except Exception as e:
            print(f"[Kafka生产者] 订单发送失败：{str(e)}")

    def close(self):
        self.producer.close()

# 测试生产者
if __name__ == "__main__":
    producer = KafkaOrderProducer()
    for i in range(3):
        order_id = f"O20250520{str(i).zfill(3)}"
        order_data = {
            "order_id": order_id,
            "user_id": "U1001",
            "amount": 199.9 + i,
            "create_time": "2025-05-20 14:00:00"
        }
        producer.send_order_message(order_data)
    producer.close()