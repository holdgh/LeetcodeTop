# rocketmq_producer.py
import json
from rocketmq.client import Producer, Message
from config import ROCKETMQ_CONFIG

class RocketMQOrderProducer:
    def __init__(self):
        self.producer = Producer(ROCKETMQ_CONFIG["producer_group"])
        self.producer.set_namesrv_addr(ROCKETMQ_CONFIG["namesrv_addr"])
        self.producer.start()

    def send_order_message(self, order_data):
        """发送订单消息"""
        msg = Message(ROCKETMQ_CONFIG["topic"])
        # 设置消息 Key（订单号，用于幂等和查询）
        msg.set_keys(order_data["order_id"])
        # 设置消息体
        msg.set_body(json.dumps(order_data, ensure_ascii=False).encode("utf-8"))
        # 同步发送消息（可靠投递）
        result = self.producer.send_sync(msg)
        print(f"[RocketMQ生产者] 订单 {order_data['order_id']} 发送成功，msg_id：{result.msg_id}")

    def close(self):
        self.producer.shutdown()

# 测试生产者
if __name__ == "__main__":
    producer = RocketMQOrderProducer()
    for i in range(3):
        order_id = f"O20250520{str(i).zfill(3)}"
        order_data = {
            "order_id": order_id,
            "user_id": "U1001",
            "amount": 299.9 + i,
            "create_time": "2025-05-20 16:00:00"
        }
        producer.send_order_message(order_data)
    producer.close()