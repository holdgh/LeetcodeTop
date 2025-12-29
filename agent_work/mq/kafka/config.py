# kafka_config.py
from kafka import KafkaProducer, KafkaConsumer

# Kafka 配置
KAFKA_CONFIG = {
    "bootstrap_servers": ["localhost:9092"],
    "topic": "order_topic",
    "consumer_group_id": "order_consumer_group"
}

# 幂等性存储（生产环境用 Redis）
processed_order_ids = set()