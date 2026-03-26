def maopao(arr):
    l = len(arr)
    if l < 2:
        return
    for i in range(l - 1, 0, -1):
        cur_max = i
        for j in range(i):
            if arr[j] > arr[cur_max]:
                cur_max = j
        arr[i], arr[cur_max] = arr[cur_max], arr[i]


def xuanze(arr):
    l = len(arr)
    if l < 2:
        return
    for i in range(l - 1):
        cur_min = i
        for j in range(i + 1, l):
            if arr[j] < arr[cur_min]:
                cur_min = j
        arr[i], arr[cur_min] = arr[cur_min], arr[i]


def charu(arr):
    l = len(arr)
    if l < 2:
        return
    for i in range(1, l):
        target_index = i
        target_value = arr[i]
        for j in range(i - 1, -1, -1):
            if arr[j] > target_value:
                arr[target_index] = arr[j]
                target_index = target_index - 1
            else:
                break
        arr[target_index] = target_value


def guibing(arr):
    l = len(arr)
    if l < 2:
        return
    mid_ind = l // 2
    left, right = arr[:mid_ind], arr[mid_ind:]
    guibing(left)
    guibing(right)
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
        k = k + 1
        i = i + 1

    while j < len(right):
        arr[k] = right[j]
        j = j + 1
        k = k + 1


def kuaisu(arr):
    def get_mid_ind(arr, low, high):
        target_value = arr[high]
        target_index = low
        for i in range(low, high):
            if arr[i] < target_value:
                arr[i], arr[target_index] = arr[target_index], arr[i]
                target_index = target_index + 1
        arr[high], arr[target_index] = arr[target_index], arr[high]
        return target_index

    def quick_sort(arr, low, high):
        if low >= high:
            return
        mid_ind = get_mid_ind(arr, low, high)
        quick_sort(arr, low, mid_ind - 1)
        quick_sort(arr, mid_ind + 1, high)

    l = len(arr)
    if l < 2:
        return
    quick_sort(arr, 0, l - 1)


def duipai(arr):
    def create_dui(arr, node_count, parent):
        cur_max = parent
        left, right = 2 * parent + 1, 2 * parent + 2
        if left < node_count and arr[left] > arr[cur_max]:
            cur_max = left
        if right < node_count and arr[right] > arr[cur_max]:
            cur_max = right
        if cur_max != parent:
            arr[parent], arr[cur_max] = arr[cur_max], arr[parent]
            create_dui(arr, node_count, cur_max)

    l = len(arr)
    if l < 2:
        return
    for i in range(l // 2 - 1, -1, -1):
        create_dui(arr, l, i)
    for j in range(l - 1, 0, -1):
        arr[j], arr[0] = arr[0], arr[j]
        create_dui(arr, j, 0)


if __name__ == '__main__':
    arr = [3, 2, 5, 4, 7, 6, 9, 8, 11, 10]
    print(f'排序前：{arr}')
    # maopao(arr)
    # xuanze(arr)
    # charu(arr)
    # guibing(arr)
    # kuaisu(arr)
    duipai(arr)
    print(f'排序后：{arr}')
