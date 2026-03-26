from multiprocessing import Process, Manager


def worker(d, l, lock):
    with lock:
        d['counter'] += 1
        l.append(d['counter'])
        print("Worker 修改数据:", d, l)


if __name__ == '__main__':
    with Manager() as manager:
        shared_dict = manager.dict({'counter': 0})
        shared_list = manager.list()
        lock = manager.Lock()

        processes = []
        for i in range(5):
            p = Process(target=worker, args=(shared_dict, shared_list, lock))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        print("最终结果:", dict(shared_dict), list(shared_list))