import signal
import os
import time
from multiprocessing import Process


def signal_handler(signum, frame):
    print(f"进程 {os.getpid()} 收到信号 {signum}")


def child_process():
    signal.signal(signal.SIGUSR1, signal_handler)
    print(f"子进程 {os.getpid()} 等待信号...")
    time.sleep(10)


if __name__ == '__main__':
    signal.signal(signal.SIGUSR1, signal_handler)

    p = Process(target=child_process)
    p.start()

    print(f"父进程 {os.getpid()} 发送信号给子进程 {p.pid}")
    os.kill(p.pid, signal.SIGUSR1)  # 发送信号

    p.join()