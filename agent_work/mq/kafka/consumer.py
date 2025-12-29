import json
import threading
import time
from kafka import KafkaConsumer
from kafka.errors import KafkaError, NoBrokersAvailable
from config import KAFKA_CONFIG

# 幂等性校验的全局集合（线程安全）
processed_order_ids = set()
# 线程锁，保证processed_order_ids的线程安全
order_lock = threading.Lock()


class KafkaOrderConsumer:
    def __init__(self, thread_count=3):
        self.thread_count = thread_count
        self.is_running = True  # 控制消费者运行状态

    def init_consumer(self):
        """初始化消费者（抽离成方法，便于异常捕获）"""
        try:
            consumer = KafkaConsumer(
                KAFKA_CONFIG["topic"],
                bootstrap_servers=KAFKA_CONFIG["bootstrap_servers"],
                group_id=KAFKA_CONFIG["consumer_group_id"],
                auto_offset_reset="earliest",  # 从最早消息开始消费
                enable_auto_commit=False,  # 关闭自动提交
                # 关键修复1：移除consumer_timeout_ms，避免线程快速退出
                # consumer_timeout_ms=1000,
                fetch_max_bytes=52428800,
                # 关键：添加值反序列化器，避免手动decode
                value_deserializer=lambda m: json.loads(m.decode('utf-8')) if m else None,
                api_version=(2, 8, 0),  # 指定Kafka版本，避免版本兼容问题
            )
            print(f"[消费者初始化] 成功连接Kafka，主题：{KAFKA_CONFIG['topic']}")
            return consumer
        except NoBrokersAvailable:
            print(f"[消费者初始化失败] 无法连接Kafka：{KAFKA_CONFIG['bootstrap_servers']}，请检查服务是否启动")
            return None
        except KafkaError as e:
            print(f"[消费者初始化失败] Kafka异常：{str(e)}")
            return None
        except Exception as e:
            print(f"[消费者初始化失败] 未知异常：{str(e)}")
            return None

    def process_order(self, consumer, thread_name):
        """消费订单消息（优化逻辑+线程安全）"""
        if not consumer:
            return

        print(f"[{thread_name}] 启动成功，开始消费消息...")
        while self.is_running:
            try:
                # 手动拉取消息（替代for循环，更可控）
                messages = consumer.poll(timeout_ms=3000)  # 3秒拉取一次
                if not messages:
                    # 无消息时打印提示，避免误以为代码没运行
                    print(f"[{thread_name}] 暂未拉取到消息，继续等待...")
                    continue

                # 遍历拉取到的消息（按分区分组）
                for partition, msgs in messages.items():
                    for msg in msgs:
                        try:
                            order_data = msg.value  # 已通过反序列化器解析
                            if not order_data or "order_id" not in order_data:
                                print(f"[{thread_name}] 消息格式错误：{msg.value}")
                                consumer.commit({partition: msg.offset + 1})  # 提交偏移量，跳过错误消息
                                continue

                            order_id = order_data["order_id"]

                            # 1. 幂等性校验（加线程锁）
                            with order_lock:
                                if order_id in processed_order_ids:
                                    print(f"[{thread_name}] 订单 {order_id} 已处理，跳过")
                                    # 手动提交偏移量（指定分区+偏移量，更可靠）
                                    consumer.commit({partition: msg.offset + 1})
                                    continue

                            # 2. 模拟业务处理
                            print(f"[{thread_name}] 处理订单：{order_id}，金额：{order_data.get('amount', 0)}")
                            time.sleep(0.5)  # 模拟业务耗时

                            # 3. 标记已处理（加线程锁）
                            with order_lock:
                                processed_order_ids.add(order_id)

                            # 4. 手动提交偏移量（关键：提交当前消息的下一个偏移量）
                            consumer.commit({partition: msg.offset + 1})
                            print(f"[{thread_name}] 订单 {order_id} 处理完成，已提交Offset：{msg.offset + 1}")

                        except Exception as e:
                            print(f"[{thread_name}] 单条消息处理失败：{str(e)}，消息：{msg.value}")
                            # 消费失败：可重试，此处简单跳过并提交偏移量
                            consumer.commit({partition: msg.offset + 1})
                            continue

            except KeyboardInterrupt:
                print(f"[{thread_name}] 接收到停止信号，准备退出...")
                self.is_running = False
            except Exception as e:
                print(f"[{thread_name}] 消费异常：{str(e)}，3秒后重试...")
                time.sleep(3)

        # 关闭消费者
        consumer.close()
        print(f"[{thread_name}] 已停止消费，关闭消费者连接")

    def start_concurrent_consume(self):
        """多线程并发消费（优化）"""
        threads = []
        for i in range(self.thread_count):
            thread_name = f"Kafka-Consumer-{i + 1}"
            # 初始化消费者
            consumer = self.init_consumer()
            if not consumer:
                print(f"[{thread_name}] 消费者初始化失败，跳过该线程")
                continue

            # 创建线程
            t = threading.Thread(
                target=self.process_order,
                args=(consumer, thread_name),
                name=thread_name,
                daemon=True  # 设为守护线程，主进程退出时自动结束
            )
            t.start()
            threads.append(t)
            print(f"[{thread_name}] 线程已启动")

        # 等待线程（防止主进程退出）
        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[主进程] 接收到停止信号，停止所有消费者线程...")
            self.is_running = False
            for t in threads:
                t.join(timeout=5)
            print("[主进程] 所有消费者线程已停止")


# 测试消费者
if __name__ == "__main__":
    print("=== Kafka消费者启动中 ===")
    consumer = KafkaOrderConsumer(thread_count=3)
    consumer.start_concurrent_consume()