# rocketmq_config.py
ROCKETMQ_CONFIG = {
    "namesrv_addr": "localhost:9876",
    "topic": "order_topic",
    "producer_group": "order_producer_group",
    "consumer_group": "order_consumer_group"
}

# 幂等性存储
processed_orders = set()