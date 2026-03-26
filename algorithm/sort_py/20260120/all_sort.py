def mao_pao(arr: list):
    n = len(arr)
    if n < 2:
        return
    for i in range(n - 1, 0, -1):
        for j in range(0, i, 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]


def xuan_ze(arr: list):
    n = len(arr)
    if n < 2:
        return
    for i in range(n):
        cur_min = i
        for j in range(i + 1, n):
            if arr[j] < cur_min:
                cur_min = j
        arr[i], arr[cur_min] = arr[cur_min], arr[i]


def cha_ru(arr: list):
    n = len(arr)
    if n < 2:
        return
    for i in range(1, n, 1):  # 待排序元素集合
        key = arr[i]  # 当前待排序元素
        j = i - 1
        while j > -1 and arr[j] > key:  # 当前已排序元素，从后往前比较
            arr[j + 1] = arr[j]  # 大于目标元素时，则后移一位
            j = j - 1  # 接着比较前一个元素
        arr[j + 1] = key


def gui_bing(arr: list):
    n = len(arr)
    if n < 2:
        return
    mid = n // 2
    left = arr[:mid]
    right = arr[mid:]
    gui_bing(left)
    gui_bing(right)
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


def kuai_pai(arr: list):
    n = len(arr)
    if n < 2:
        return

    def quick_sort(arr: list, low, high):
        if low >= high:
            return

        def get_mid(arr: list, low, high):
            mid = low
            mid_value = arr[high]
            for i in range(low, high):
                if arr[i] < mid_value:
                    arr[i], arr[mid] = arr[mid], arr[i]
                    mid = mid + 1
            arr[mid], arr[high] = arr[high], arr[mid]
            return mid

        mid = get_mid(arr, low, high)
        quick_sort(arr, low, mid - 1)
        quick_sort(arr, mid + 1, high)

    quick_sort(arr, 0, n - 1)


def dui_pai(arr: list):
    n = len(arr)
    if n < 2:
        return
    def heapify(arr, node_count, root):
        cur_max = root
        left = 2*root+1
        right = 2*root+2
        if left < node_count and arr[left] > arr[cur_max]:
            cur_max = left
        if right < node_count and arr[right] > arr[cur_max]:
            cur_max = right
        if cur_max != root:
            arr[root], arr[cur_max] = arr[cur_max], arr[root]
            heapify(arr, node_count, cur_max)

    def build_max_heap(arr: list):
        n = len(arr)
        if n < 2:
            return
        for i in range(n // 2 - 1, -1 , -1):
            heapify(arr, n, i)

    build_max_heap(arr)
    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)


if __name__ == '__main__':
    arr = [9, 8, 10, 15, 6, 7, 3, 45, 6, 11, 18, 52, 147]
    print(f"排序前：{arr}")
    # mao_pao(arr)
    # xuan_ze(arr)
    # cha_ru(arr)
    # gui_bing(arr)
    # kuai_pai(arr)
    dui_pai(arr)
    print(f"排序后：{arr}")
