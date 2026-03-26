def kuai_pai(arr):
    def quick_sort(arr, low, high):
        if low >= high:
            return

        def get_mid_index(arr, low, high):
            mid_index = low
            for i in range(low, high):
                if arr[i] < arr[high]:
                    if i > mid_index:
                        arr[i], arr[mid_index] = arr[mid_index], arr[i]
                    mid_index = mid_index + 1
            arr[mid_index], arr[high] = arr[high], arr[mid_index]
            return mid_index

        mid_index = get_mid_index(arr, low, high)
        quick_sort(arr, low, mid_index - 1)
        quick_sort(arr, mid_index + 1, high)

    l = len(arr)
    if l < 2:
        return
    quick_sort(arr, 0, l - 1)


def xuan_ze(arr):
    l = len(arr)
    if l < 2:
        return
    for i in range(1, l):
        target_value = arr[i]
        target_index = i
        for j in range(i - 1, -1, -1):
            if arr[j] > target_value:
                arr[target_index] = arr[j]
                target_index = target_index - 1
            else:
                break
        arr[target_index] = target_value


def dui_pai(arr):
    def create_max_dui(arr, node_count, parent):
        cur_max = parent
        left, right = 2 * parent + 1, 2 * parent + 2
        if left < node_count and arr[left] > arr[cur_max]:
            cur_max = left
        if right < node_count and arr[right] > arr[cur_max]:
            cur_max = right
        if cur_max != parent:
            arr[cur_max], arr[parent] = arr[parent], arr[cur_max]
            create_max_dui(arr, node_count, cur_max)

    l = len(arr)
    if l < 2:
        return
    for i in range(l // 2 - 1, -1, -1):
        create_max_dui(arr, l, i)
    for j in range(l - 1, 0, -1):
        arr[j], arr[0] = arr[0], arr[j]
        create_max_dui(arr, j, 0)


if __name__ == '__main__':
    arr = [3, 2, 5, 4, 7, 6, 9, 8, 11, 10]
    print(f'排序前：{arr}')
    kuai_pai(arr)
    # xuan_ze(arr)
    # dui_pai(arr)
    print(f'排序后：{arr}')
