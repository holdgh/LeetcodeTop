# 服务器端
from xmlrpc.server import SimpleXMLRPCServer
from multiprocessing import Process


class RPCService:
    def say_hello(self, name):
        return f"Hello, {name}!"


def rpc_server():
    server = SimpleXMLRPCServer(('localhost', 8000))
    server.register_instance(RPCService())
    print("RPC 服务器运行在 http://localhost:8000")
    server.serve_forever()


# 客户端
import xmlrpc.client


def rpc_client():
    proxy = xmlrpc.client.ServerProxy("http://localhost:8000/")  # 设置服务端连接地址
    print(proxy.say_hello("World"))


if __name__ == '__main__':
    server = Process(target=rpc_server)  # 创建一个进程，执行服务端操作
    server.start()

    # 给服务器一点启动时间
    import time

    time.sleep(1)

    client = Process(target=rpc_client)  # 创建一个进程，执行客户端操作
    client.start()

    client.join()
    server.terminate()  # 终止服务器进程