def gui_bing(arr):
    l = len(arr)
    if l < 2:
        return
    mid = l // 2
    left, right = arr[:mid], arr[mid:]
    # 递归
    gui_bing(left)
    gui_bing(right)
    # 合并两个有序列表
    i, j, k = 0, 0, 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            arr[k] = left[i]
            i = i + 1
        else:
            arr[k] = right[j]
            j = j + 1
        k = k + 1
    while i < len(left):
        arr[k] = left[i]
        i = i + 1
        k = k + 1
    while j < len(right):
        arr[k] = right[j]
        j = j + 1
        k = k + 1


if __name__ == '__main__':
    arr = [3, 2, 5, 4, 7, 6]
    print(f'排序前：{arr}')
    gui_bing(arr)
    print(f'排序后：{arr}')
