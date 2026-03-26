def kuai_su(arr):
    def quick_sort(arr, low, high):
        if low >= high:
            return
        pivot_index = get_pivot_index(arr, low, high)
        quick_sort(arr, low, pivot_index - 1)
        quick_sort(arr, pivot_index + 1, high)

    def get_pivot_index(arr, low, high):
        pivot_value = arr[high]
        pivot_index = low
        for i in range(low, high):
            if arr[i] < pivot_value:
                arr[i], arr[pivot_index] = arr[pivot_index], arr[i]
                pivot_index = pivot_index + 1
        arr[pivot_index], arr[high] = arr[high], arr[pivot_index]
        return pivot_index

    l = len(arr)
    if l < 2:
        return
    quick_sort(arr, 0, l - 1)


def dui_pai(arr):
    def create_da_ding_dui(arr, node_count, parent):
        cur_max = parent
        left, right = 2 * parent + 1, 2 * parent + 2
        if left < node_count and arr[left] > arr[cur_max]:
            cur_max = left
        if right < node_count and arr[right] > arr[cur_max]:
            cur_max = right
        if cur_max != parent:
            arr[parent], arr[cur_max] = arr[cur_max], arr[parent]
            create_da_ding_dui(arr, node_count, cur_max)

    l = len(arr)
    if l < 2:
        return
    for i in range(l // 2 - 1, -1, -1):
        create_da_ding_dui(arr, l, i)
    for j in range(l - 1, 0, -1):
        arr[j], arr[0] = arr[0], arr[j]
        create_da_ding_dui(arr, j, 0)


if __name__ == '__main__':
    arr = [3, 2, 5, 4, 7, 6, 9, 8, 11, 10]
    print(f'排序前：{arr}')
    # kuai_su(arr)
    dui_pai(arr)
    print(f'排序后：{arr}')
