# config.py
import pika

# RabbitMQ 连接配置
RABBITMQ_CONFIG = {
    "host": "localhost",
    "port": 5672,
    "username": "guest",
    "password": "guest",
    "queue_name": "order_queue",  # 订单队列名
    "exchange_name": "order_exchange",  # 订单交换机
    "routing_key": "order.key"
}

# 幂等性存储（生产环境建议用 Redis）
order_id_set = set()