from multiprocessing import Process, Pipe


def child_process(conn):
    print("子进程收到消息:", conn.recv())  # 接收消息
    conn.send("Hello Parent!")  # 发送消息
    conn.close()


if __name__ == '__main__':
    """
    管道是半双工的通信方【支撑双向通信，但是不能同时收发】，适用于父子进程或兄弟进程之间的通信。
    """
    parent_conn, child_conn = Pipe()  # 创建管道，本质是multiprocessing.connection.Connection二元组
    p = Process(target=child_process, args=(child_conn,))  # 创建一个进程实例【相对于“if __name__ == '__main__':”的MainProcess，该进程实例为子进程】
    p.start()

    parent_conn.send("Hello Child!")  # 发送消息
    print("父进程收到消息:", parent_conn.recv())  # 接收消息

    p.join()