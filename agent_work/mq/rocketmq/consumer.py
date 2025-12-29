# rocketmq_consumer.py
import json
import threading
from rocketmq.client import PushConsumer, ConsumeStatus
from config import ROCKETMQ_CONFIG, processed_orders


class RocketMQOrderConsumer:
    def __init__(self, thread_count=3):
        self.thread_count = thread_count

    def consume_message(self, msg):
        """消费回调函数"""
        try:
            order_data = json.loads(msg.body.decode("utf-8"))
            order_id = order_data["order_id"]

            # 幂等性校验
            if order_id in processed_orders:
                print(f"[RocketMQ消费者] 订单 {order_id} 已处理，跳过")
                return ConsumeStatus.CONSUME_SUCCESS  # 返回成功，不重试

            # 模拟业务处理
            print(f"[RocketMQ消费者] 处理订单：{order_id}，金额：{order_data['amount']}")
            # TODO: 核心业务逻辑

            # 标记已处理
            processed_orders.add(order_id)

            print(f"[RocketMQ消费者] 订单 {order_id} 处理完成")
            return ConsumeStatus.CONSUME_SUCCESS  # 消费成功

        except Exception as e:
            print(f"[RocketMQ消费者] 订单处理失败：{str(e)}")
            return ConsumeStatus.RECONSUME_LATER  # 稍后重试

    def start_consumer(self):
        """启动单个消费者"""
        consumer = PushConsumer(ROCKETMQ_CONFIG["consumer_group"])
        consumer.set_namesrv_addr(ROCKETMQ_CONFIG["namesrv_addr"])
        # 订阅 Topic，* 表示订阅所有 Tag
        consumer.subscribe(ROCKETMQ_CONFIG["topic"], "*", self.consume_message)
        # 设置消费模式：集群消费（默认）
        consumer.set_message_model("CLUSTERING")
        # 启动消费者
        consumer.start()
        print(f"[RocketMQ消费者] 启动成功，等待消息...")
        # 保持线程存活
        while True:
            pass

    def start_concurrent_consume(self):
        """多线程并发消费"""
        threads = []
        for i in range(self.thread_count):
            t = threading.Thread(target=self.start_consumer, name=f"RocketMQ-Consumer-{i + 1}")
            t.start()
            threads.append(t)

        for t in threads:
            t.join()


# 测试消费者
if __name__ == "__main__":
    consumer = RocketMQOrderConsumer(thread_count=3)
    consumer.start_concurrent_consume()