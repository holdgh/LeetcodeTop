def kuai_su(arr):
    l = len(arr)
    if l < 2:
        return
    sort(arr, 0, l - 1)


def sort(arr, low, high):
    if low >= high:
        return
    mid = get_mid(arr, low, high)
    sort(arr, low, mid - 1)
    sort(arr, mid + 1, high)


def get_mid(arr, low, high):
    pivot = arr[high]
    i = low
    for j in range(low, high):
        if arr[j] < pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i = i + 1
    arr[i], arr[high] = arr[high], arr[i]
    return i


if __name__ == '__main__':
    arr = [3, 2, 5, 4, 7, 6]
    print(f'排序前：{arr}')
    kuai_su(arr)
    print(f'排序后：{arr}')