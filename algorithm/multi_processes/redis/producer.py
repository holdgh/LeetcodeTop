# 生产者进程 producer.py
import redis
import time

r = redis.Redis(host='localhost', port=6379, db=0)
if __name__ == '__main__':
    for i in range(5):
        r.rpush('myqueue', f"消息 {i}")
        print(f"生产者发送: 消息 {i}")
        time.sleep(0.5)
    r.rpush('myqueue', 'END')