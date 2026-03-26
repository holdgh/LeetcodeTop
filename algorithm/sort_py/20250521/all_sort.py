def mao_pao(arr):
    l = len(arr)
    if l < 2:
        return
    for i in range(l - 1, 0, -1):  # 未排序末尾元素遍历
        cur_max = i  # 初始化当前未排序部分的最大值索引
        for j in range(i):
            if arr[j] > arr[i]:
                cur_max = j  # 更新未排序部分的最大值索引
        arr[i], arr[cur_max] = arr[cur_max], arr[i]  # 将当前未排序部分的最大值更新至末尾


def xuan_ze(arr):
    l = len(arr)
    if l < 2:
        return
    for i in range(l - 1):  # 未排序起始元素遍历
        cur_min = i  # 初始当前未排序部分的最小值索引
        for j in range(i + 1, l):
            if arr[j] < arr[i]:
                cur_min = j  # 更新未排序部分的最小值索引
        arr[i], arr[cur_min] = arr[cur_min], arr[i]  # 将当前未排序部分的最小值更新至起始位置


def cha_ru(arr):
    l = len(arr)
    if l < 2:
        return
    for i in range(1, l):  # 从前往后遍历未排序元素
        cur_value = arr[i]  # 暂存当前未排序元素的数值
        cur_target = i  # 初始化当前未排序元素在已排序部分的目标位置
        for j in range(i - 1, -1, -1):  # 从后往前遍历已排序部分，找到正确的目标位置
            if arr[j] > cur_value:  # 已排序元素大于当前未排序元素的数值时，将当前已排序元素后移一位，同时更新目标位置往前一位
                arr[j + 1] = arr[j]  # 后移操作
                cur_target = cur_target - 1  # 目标位置前移一位
            else:  # 目标位置条件：从后往前，第一次出现已排序元素不大于当前未排序元素
                break
        arr[cur_target] = cur_value  # 将当前未排序元素放置在目标位置


def kuai_su(arr):
    def quick_sort(arr, low, high):
        if low >= high:
            return
        mid = get_mid_idx(arr, low, high)  # 以arr[high]为临界值，获取最终临界值索引
        quick_sort(arr, low, mid - 1)  # 分治，左侧
        quick_sort(arr, mid + 1, high)  # 分治，右侧

    def get_mid_idx(arr, low, high):
        pivot = arr[high]
        mid_idx = low  # 初始化临界位置
        for i in range(low, high):
            if arr[i] < pivot:  # 将小于临界值的元素左移
                arr[i], arr[mid_idx] = arr[mid_idx], arr[i]
                mid_idx = mid_idx + 1  # 更新临界位置
        arr[mid_idx], arr[high] = arr[high], arr[mid_idx]  # 将临界值放在临界位置
        return mid_idx

    l = len(arr)
    if l < 2:
        return
    quick_sort(arr, 0, l - 1)


def dui_pai(arr):
    def create_dui(arr, size, parent):
        cur_max = parent  # 暂存父节点和左右子节点的最大值索引为父节点索引
        left, right = 2 * parent + 1, 2 * parent + 2  # 获取左右子节点索引
        if left < size and arr[left] > arr[parent]:  # 存在左子节点且左子节点大于父节点时
            cur_max = left  # 更新最大值索引为左子节点索引
        if right < size and arr[right] > arr[cur_max]:  # 存在右子节点且右子节点大于最大值索引处的值时
            cur_max = right  # 更新最大值索引为右子节点索引
        if cur_max != parent:  # 最大值节点不是父节点时
            arr[parent], arr[cur_max] = arr[cur_max], arr[parent]  # 将最大值节点与父节点的数值互换
            create_dui(arr, size, cur_max)  # 处理受影响的子节点cur_max

    l = len(arr)
    if l < 2:
        return
    # 建立大顶堆。必须反向构建【小顶堆也是反向构建】。从最后一个非叶子节点【由节点个数和非叶子的左右子节点可得最后一个非叶子节点的索引为l//2-1】开始往根节点方式构建
    for i in range(l // 2 - 1, -1, -1):
        create_dui(arr, l, i)
    # 冒泡思想，排序，重构大顶堆【获取根节点，最大值】
    for j in range(l - 1, 0, -1):
        arr[j], arr[0] = arr[0], arr[j]
        create_dui(arr, j, 0)


def gui_bing(arr):
    l = len(arr)
    if l < 2:
        return
    mid = l // 2  # 初始当前分界线索引
    # 拆分左右两部分
    left = arr[:mid]
    right = arr[mid:]
    # 递归拆分
    gui_bing(left)
    gui_bing(right)
    # 合并处理左右有序子组
    i, j, k = 0, 0, 0
    while i < len(left) and j < len(right):  # 左右子组尚未处理完毕
        if left[i] < right[j]:  # 取其小者，并更新被取者索引和取者索引
            arr[k] = left[i]
            i = i + 1
        else:
            arr[k] = right[j]
            j = j + 1
        k = k + 1
    while i < len(left):  # 仅剩左子组，直接收集
        arr[k] = left[i]
        i = i + 1
        k = k + 1
    while j < len(right):  # 仅剩右子组，直接收集
        arr[k] = right[j]
        j = j + 1
        k = k + 1


if __name__ == '__main__':
    arr = [3, 2, 5, 4, 7, 6, 9, 8, 11, 10]
    print(f'排序前：{arr}')
    # mao_pao(arr)
    # xuan_ze(arr)
    # cha_ru(arr)
    # kuai_su(arr)
    # dui_pai(arr)
    gui_bing(arr)
    print(f'排序后：{arr}')
