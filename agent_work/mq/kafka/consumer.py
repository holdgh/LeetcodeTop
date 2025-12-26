# kafka_consumer.py
import json
import threading
from config import KAFKA_CONFIG, processed_order_ids
from kafka import KafkaConsumer


class KafkaOrderConsumer:
    def __init__(self, thread_count=3):
        self.thread_count = thread_count

    def process_order(self, consumer):
        """消费订单消息"""
        for message in consumer:
            try:
                order_data = json.loads(message.value.decode("utf-8"))
                order_id = order_data["order_id"]

                # 1. 幂等性校验
                if order_id in processed_order_ids:
                    print(f"[Kafka消费者] 订单 {order_id} 已处理，跳过")
                    # 手动提交 Offset，确认消息已消费
                    consumer.commit()
                    continue

                # 2. 模拟业务处理
                print(f"[Kafka消费者] 处理订单：{order_id}，金额：{order_data['amount']}")
                # TODO: 库存扣减、物流通知等业务逻辑

                # 3. 标记已处理
                processed_order_ids.add(order_id)

                # 4. 手动提交 Offset（关键：确认消息消费成功）
                consumer.commit()
                print(f"[Kafka消费者] 订单 {order_id} 处理完成，已提交 Offset")

            except Exception as e:
                print(f"[Kafka消费者] 订单处理失败：{str(e)}")
                # 消费失败：可根据需求重试或记录日志
                continue

    def start_concurrent_consume(self):
        """多线程并发消费"""
        threads = []
        for i in range(self.thread_count):
            # 每个线程创建独立的消费者实例
            consumer = KafkaConsumer(
                KAFKA_CONFIG["topic"],
                bootstrap_servers=KAFKA_CONFIG["bootstrap_servers"],
                group_id=KAFKA_CONFIG["consumer_group_id"],
                auto_offset_reset="earliest",  # 从最早的消息开始消费
                enable_auto_commit=False,  # 核心：关闭自动提交 Offset
                consumer_timeout_ms=1000,
                fetch_max_bytes=52428800  # 每次拉取的最大字节数
            )
            t = threading.Thread(target=self.process_order, args=(consumer,), name=f"Kafka-Consumer-{i + 1}")
            t.start()
            threads.append(t)

        for t in threads:
            t.join()


# 测试消费者
if __name__ == "__main__":
    consumer = KafkaOrderConsumer(thread_count=3)
    consumer.start_concurrent_consume()