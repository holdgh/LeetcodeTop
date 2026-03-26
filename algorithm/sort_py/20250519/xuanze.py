def xuan_ze(arr):
    l = len(arr)
    if l < 2:
        return
    for i in range(l):  # 从未排序元素中寻找最小值，并将最小值放置到起始位置，更新未排序的起始位置
        cur_min = i
        for j in range(i+1, l):  # 从未排序元素中寻找最小值索引
            if arr[j] < arr[cur_min]:
                cur_min = j
        arr[i], arr[cur_min] = arr[cur_min], arr[i]  # 将最小值置于起始位置


if __name__ == '__main__':
    arr = [3, 2, 5, 4, 7, 6, 9, 8]
    print(f'排序前：{arr}')
    xuan_ze(arr)
    print(f'排序后：{arr}')