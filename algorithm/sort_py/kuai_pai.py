def kuai_su(arr):
    l = len(arr)
    if l < 2:
        return
    sort(arr, 0, l - 1)


def sort(arr, low, high):
    if low < high:  # 只有一个元素时，不进行排序操作
        mid_pos = get_mid_pos(arr, low, high)  # 设置好分界值的位置并返回其分界值索引
        sort(arr, 0, mid_pos - 1)  # 递归处理分界值左侧【小于分界值的部分】
        sort(arr, mid_pos + 1, high)  # 递归处理分界值右侧【大于分界值的部分】


def get_mid_pos(arr, low, high):
    mid_value = arr[high]
    mid_pos = low
    for i in range(low, high):
        if arr[i] < mid_value:
            less_mid_value = arr[i]
            arr[i] = arr[mid_pos]
            arr[mid_pos] = less_mid_value
            mid_pos = mid_pos + 1
    arr[high] = arr[mid_pos]
    arr[mid_pos] = mid_value
    return mid_pos


if __name__ == '__main__':
    arr = [5, 3, 9, 10, 4, 7]
    print(arr)
    kuai_su(arr)
    print(arr)