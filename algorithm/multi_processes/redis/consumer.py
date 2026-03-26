# 消费者进程 consumer.py
import redis

r = redis.Redis(host='localhost', port=6379, db=0)
if __name__ == '__main__':
    while True:
        msg = r.blpop('myqueue')[1].decode()
        if msg == "END":
            break
        print(f"消费者接收: {msg}")