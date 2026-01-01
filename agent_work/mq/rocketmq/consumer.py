# rocketmq_consumer.py
from config import ROCKETMQ_CONFIG, processed_orders
import time
from typing import List

from pyrocketmq.client.consumer.listener import MessageListenerConcurrently, ConsumeConcurrentlyContext, \
    ConsumeConcurrentlyStatus
from pyrocketmq.common.message import MessageExt

from enum import Enum

from pyrocketmq.common.common import MessageModel, ConsumeFromWhere

import json
import threading
from pyrocketmq.client.consumer.consumer import PushConsumer, MessageSelector


class MyMessageListenerConcurrently(MessageListenerConcurrently):
    def _consumeMessage(self, msgs: List[MessageExt],
                        context: ConsumeConcurrentlyContext) -> ConsumeConcurrentlyStatus:  # 自定义消费逻辑
        print('Concurrently', context.ackIndex)
        for msg in msgs:
            try:
                order_data = json.loads(msg.body.decode("utf-8"))
                order_id = order_data["order_id"]

                # 幂等性校验
                if order_id in processed_orders:
                    print(f"[RocketMQ消费者] 订单 {order_id} 已处理，跳过")

                # 模拟业务处理
                print(f"[RocketMQ消费者] 处理订单：{order_id}，金额：{order_data['amount']}")
                # TODO: 核心业务逻辑

                # 标记已处理
                processed_orders.add(order_id)

                print(f"[RocketMQ消费者{threading.current_thread().name}] 订单 {order_id} 处理完成")

            except Exception as e:
                print(f"[RocketMQ消费者] 订单处理失败：{str(e)}")
            print(json.loads(msg.body))
        return ConsumeConcurrentlyStatus.CONSUME_SUCCESS


class RocketMQOrderConsumer:
    def __init__(self, thread_count=3):
        self.thread_count = thread_count

    def start_consumer(self):
        """启动单个消费者"""
        cs = PushConsumer(ROCKETMQ_CONFIG["consumer_group"])
        cs.setNamesrvAddr(ROCKETMQ_CONFIG["namesrv_addr"])
        selector = MessageSelector.byTag('order')
        ml = MyMessageListenerConcurrently()  # 关键逻辑：自定义消费逻辑
        cs.registerMessageListener(ml)
        # 关键逻辑：基于selector订阅特定标志的消息
        cs.subscribe(ROCKETMQ_CONFIG["topic"], selector)
        cs.setConsumeFromWhere(ConsumeFromWhere.CONSUME_FROM_FIRST_OFFSET)
        cs.start()
        print(f"[RocketMQ消费者{threading.current_thread().name}] 启动成功，等待消息...")
        # 保持线程存活
        while True:
            pass
        # time.sleep(5)
        # cs.shutdown()

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