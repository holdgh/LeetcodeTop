def kuai_pai(arr: list):
    n = len(arr)
    if n < 2:
        return

    def quick_sort(arr: list, low, high):
        if low >= high:
            return

        def get_mid(arr, low, high):
            mid_value = arr[high]
            mid_ind = low
            for i in range(low, high):
                if arr[i] < mid_value:
                    arr[i], arr[mid_ind] = arr[mid_ind], arr[i]
                    mid_ind = mid_ind + 1
            arr[mid_ind], arr[high] = arr[high], arr[mid_ind]
            return mid_ind

        mid_ind = get_mid(arr, low, high)
        quick_sort(arr, low, mid_ind - 1)
        quick_sort(arr, mid_ind + 1, high)

    quick_sort(arr, 0, n-1)


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


def dui_pai(arr: list):
    n = len(arr)
    if n < 2:
        return

    def heapify(arr: list, node_count, root):
        cur_max = root
        left = 2 * root + 1
        right = 2 * root + 2
        if left < node_count and arr[left] > arr[cur_max]:
            cur_max = left
        if right < node_count and arr[right] > arr[cur_max]:
            cur_max = right
        if cur_max != root:
            arr[root], arr[cur_max] = arr[cur_max], arr[root]
            heapify(arr, node_count, cur_max)

    def build_max_heap(arr: list):
        n = len(arr)
        for i in range(n // 2 - 1, -1, -1):
            heapify(arr, n, i)

    build_max_heap(arr)

    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)


if __name__ == '__main__':
    arr = [5, 9, 3, 7, 13, 18, 4, 6, 1]
    print(f"排序前：{arr}")
    # gui_bing(arr)
    # dui_pai(arr)
    kuai_pai(arr)
    print(f"排序后：{arr}")
