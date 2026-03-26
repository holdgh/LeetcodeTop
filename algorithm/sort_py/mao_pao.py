def mao_pao(arr):
    l = len(arr)
    if l < 2:
        return
    for i in range(l):
        # 从当前状态的第1个，第2个，……，第l-i个元素中寻找最大值，并将其置于第l-i个位置上
        for j in range(l - i - 1):  # 这里减1的目的在于循环体中获取当前索引的下一个元素
            if arr[j] > arr[j+1]:  # 比较相邻元素
                # 将较大值往后移
                tmp = arr[j]
                arr[j] = arr[j+1]
                arr[j+1] = tmp


if __name__ == '__main__':
    arr = [5, 3, 9, 10, 4, 7]
    print(arr)
    mao_pao(arr)
    print(arr)