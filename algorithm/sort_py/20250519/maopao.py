def mao_pao(arr):
    l = len(arr)
    if l < 2:
        return
    for i in range(l-1, 0, -1):  # 从未排序元素中寻找最大值，并将最大值放置到最后，更新未排序的元素长度
        cur_max = i
        for j in range(i):  # 从未排序元素中寻找最大值索引
            if arr[j] > arr[cur_max]:
                cur_max = j
        arr[i], arr[cur_max] = arr[cur_max], arr[i]


if __name__ == '__main__':
    arr = [3, 2, 5, 4, 7, 6, 9, 8]
    print(f'排序前：{arr}')
    mao_pao(arr)
    print(f'排序后：{arr}')