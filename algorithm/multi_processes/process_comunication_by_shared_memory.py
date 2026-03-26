from multiprocessing import Process, Value, Array


def modify_shared_data(n, a):
    n.value = 3.1415927
    for i in range(len(a)):
        a[i] = -a[i]


if __name__ == '__main__':
    num = Value('d', 0.0)  # 'd' 表示双精度浮点数
    arr = Array('i', range(10))  # 'i' 表示有符号整数

    print("初始值:", num.value, list(arr))

    p = Process(target=modify_shared_data, args=(num, arr))
    p.start()
    p.join()

    print("修改后:", num.value, list(arr))