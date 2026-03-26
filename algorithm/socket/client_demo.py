import socket
import threading


def receive_messages(sock):
    while True:
        try:
            data = sock.recv(1024).decode('utf-8')
            if not data:
                break
            print(f"[来自服务器]: {data}")
        except ConnectionResetError:
            print("[连接已断开]")
            break


def start_client():
    host = '127.0.0.1'
    port = 65432

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        print(f"[已连接] 连接到服务器 {host}:{port}")

        # 启动接收消息的线程
        receive_thread = threading.Thread(target=receive_messages, args=(s,))
        receive_thread.daemon = True
        receive_thread.start()

        try:
            while True:
                message = input("输入消息: ")
                s.send(message.encode('utf-8'))
        except KeyboardInterrupt:
            print("\n[断开连接]")
        finally:
            s.close()


if __name__ == "__main__":
    start_client()