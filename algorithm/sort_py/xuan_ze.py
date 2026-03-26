def xuan_ze(arr):
    l = len(arr)
    if l < 2:
        return
    for i in range(l):  # 未排序元素的起始索引
        min_idx = i  # 假定未排序的第一个元素为最小值
        for j in range(i+1, l):  # 从未排序的元素中找出最小值【相邻比较，更新最小值位置】
            if arr[j] < arr[min_idx]:
                min_idx = j  # 更新最小值位置
        tmp = arr[i]  # 记录假定的最小值
        arr[i] = arr[min_idx]  # 更新真实的最小值【随着i的遍历，这里表示了已排序情况，arr[i]在这里代表了已排序的末尾】
        arr[min_idx] = tmp  # 将假定的最小值保存到真实最小值的位置【用于寻找后续批次的最小值】


if __name__ == '__main__':
    arr = [5, 3, 9, 10, 4, 7]
    print(arr)
    xuan_ze(arr)
    print(arr)