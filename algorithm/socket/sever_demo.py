import socket
import threading


def handle_client(conn, addr):
    print(f"[新连接] {addr} 已连接")

    try:
        while True:
            # 接收消息
            data = conn.recv(1024).decode('utf-8')
            if not data:
                break
            print(f"[来自 {addr}]: {data}")

            # 发送消息
            message = input("回复消息: ")
            conn.send(message.encode('utf-8'))
    except ConnectionResetError:
        print(f"[连接断开] {addr} 已断开连接")
    finally:
        conn.close()


def start_server():
    host = '127.0.0.1'
    port = 65432

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print(f"[服务器启动] 监听 {host}:{port}")

        while True:
            conn, addr = s.accept()
            client_thread = threading.Thread(target=handle_client, args=(conn, addr))
            client_thread.start()


if __name__ == "__main__":
    start_server()