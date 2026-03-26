# 服务器端
import socket
from multiprocessing import Process


def handle_client(conn, addr):
    print(f"连接来自 {addr}")
    with conn:
        while True:
            data = conn.recv(1024)
            if not data:
                break
            print(f"收到: {data.decode()}")
            conn.sendall(data)  # 回显


def socket_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 65432))
        s.listen()
        print("服务器启动，等待连接...")
        while True:
            conn, addr = s.accept()
            p = Process(target=handle_client, args=(conn, addr))
            p.start()


# 客户端
def socket_client():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('localhost', 65432))
        s.sendall(b"Hello, Socket!")
        data = s.recv(1024)
        print(f"收到回显: {data.decode()}")


if __name__ == '__main__':
    server = Process(target=socket_server)
    server.start()

    # 给服务器一点启动时间
    import time

    time.sleep(1)

    client = Process(target=socket_client)
    client.start()

    client.join()
    server.terminate()  # 终止服务器进程