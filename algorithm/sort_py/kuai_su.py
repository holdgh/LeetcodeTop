def quick_sort(arr, low, high):
    if low < high:
        pivot_idx = partition(arr, low, high)  # 得到分隔位置，此时arr[pivot_idx]处于最终排序结果的正确位置
        quick_sort(arr, low, pivot_idx - 1)  # 处理pivot_idx之前的数组
        quick_sort(arr, pivot_idx + 1, high)  # 处理pivot_idx之后的数组


def partition(arr, low, high):
    pivot = arr[high]  # 目标分界值
    i = low  # 初始化目标分界点
    for j in range(low, high):  # 遍历当前范围的数组元素，寻找目标分界点位置。在这一过程中，将所有比pivot小的值都放到目标分界点的左边
        if arr[j] < pivot:  # 当前位置的值小于目标分界值，则将其与当前的目标分界点互换位置【保证小于目标分界值的元素位于目标分界点的左侧】，同时更新目标分界点
            arr[i], arr[j] = arr[j], arr[i]  # 将其与当前的目标分界点互换位置【保证小于目标分界值的元素位于目标分界点的左侧】
            i = i + 1  # 更新目标分界点
    arr[i], arr[high] = arr[high], arr[i]  # 将目标分界值与找到的目标分界点互换，使得目标分界值位于正确的位置，同时原目标节点的值【不小于目标分界值】位于目标分界点的右侧
    return i  # 目标分界点【此处的元素是最终排序结果的正确位置】


if __name__ == '__main__':
    arr = [5, 3, 9, 10, 4, 7]
    print(arr)
    quick_sort(arr, 0, len(arr)-1)
    print(arr)
