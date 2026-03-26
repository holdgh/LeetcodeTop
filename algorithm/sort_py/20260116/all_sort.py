# 将剩余元素最大值移至末尾
def mao_pao(arrs: list):
    arr_len = len(arrs)
    if arr_len < 2:
        return
    for i in range(arr_len):
        for j in range(arr_len - i - 1):
            if arrs[j] > arrs[j + 1]:
                tmp = arrs[j]
                arrs[j] = arrs[j + 1]
                arrs[j + 1] = tmp


# 将后面未排序的元素插入到前面已排序元素的合适位置[从后往前，第一个不大于未排序元素的元素位置后一个]
def cha_ru(arrs: list):
    arr_len = len(arrs)
    if arr_len < 2:
        return
    for i in range(1, arr_len):
        key = arrs[i]
        j = i - 1  # 初始化目标位置为已排序元素的末尾，j表示目标位置的前一个位置
        while j > -1 and key < arrs[j]:
            arrs[j + 1] = arrs[j]  # 将有序元素arrs[j]后移一位，挪位置
            j = j - 1  # 对应将目标位置前移一位，更新目标位置
        arrs[j + 1] = key  # [从后往前，第一个不大于未排序元素的元素位置后一个]


# 将剩余元素的最小值移至开头
# def xuan_ze(arrs: list):
#     arr_len = len(arrs)
#     if arr_len < 2:
#         return
#     for i in range(arr_len):
#         for j in range(arr_len-1, i, -1):
#             if arrs[j-1] > arrs[j]:
#                 tmp = arrs[j-1]
#                 arrs[j-1] = arrs[j]
#                 arrs[j] = tmp
def xuan_ze(arrs: list):
    arr_len = len(arrs)
    if arr_len < 2:
        return
    for i in range(arr_len):  # 从前往后寻找剩余元素的最小值，并前移
        cur_min = i  # 假定的当前剩余元素的最小值索引
        # 寻找当前剩余元素的真实最小值索引
        for j in range(i+1, arr_len):
            if arrs[j] < arrs[cur_min]:
                cur_min = j
        # 将真实最小值与假定的最小值互换位置，相当于将真实最小值前移
        tmp = arrs[i]
        arrs[i] = arrs[cur_min]
        arrs[cur_min] = tmp


# 分治思想：重复【找临界值，将临界值放置在合适位置，使得其左侧皆小于临界值，右侧皆不小于临界值】，直至左侧和右侧仅剩一个元素【low>=high】
def kuai_pai(arrs: list):

    arr_len = len(arrs)
    if arr_len < 2:
        return

    def quick_sort(arrs: list, low, high):  # 快排算法：拟定临界值，将当前元素分为左右两部分，左部分皆小于临界值，右部分皆不小于临界值。再对左右部分重复进行相同操作，直至每部分仅有一个元素

        def get_mid(arrs: list, low, high):  # 找临界值的最终位置
            mark_value = arrs[high]
            mark_ind = low
            for i in range(low, high):
                if arrs[i] < mark_value:
                    arrs[i], arrs[mark_ind] = arrs[mark_ind], arrs[i]
                    mark_ind = mark_ind + 1
            arrs[mark_ind], arrs[high] = arrs[high], arrs[mark_ind]
            return mark_ind

        if low < high:
            mid = get_mid(arrs, low, high)  # 找到临界值的最终位置
            quick_sort(arrs, low, mid-1)  # 左部分递归进行相同处理
            quick_sort(arrs, mid+1, high)  # 右部分递归进行相同处理

    quick_sort(arrs, 0, arr_len-1)  # 对0和arr_len-1范围的元素进行快速排序


# 分治思想，先由长到短拆分到不可分，再由短到长有序合并
def gui_bing(arrs: list):
    arr_len = len(arrs)
    if arr_len < 2:
        return
    # 划分
    mid = arr_len // 2
    left = arrs[:mid]
    right = arrs[mid:]
    # 递归划分
    gui_bing(left)
    gui_bing(right)
    # 有序合并
    i = 0
    j = 0
    k = 0
    # 左右子组皆未合并完毕
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            arrs[k] = left[i]
            i = i + 1
        else:
            arrs[k] = right[j]
            j = j + 1
        k = k + 1
    # 仅左子组未合并完毕
    while i < len(left):
        arrs[k] = left[i]
        i = i + 1
        k = k + 1
    # 仅右子组未合并完毕
    while j < len(right):
        arrs[k] = right[j]
        j = j + 1
        k = k + 1


def dui_pai(arrs: list):
    arr_len = len(arrs)
    if arr_len < 2:
        return

    def heapify(arr, node_count, root):
        cur_max = root  # 最大值位置
        left = 2*root+1  # 左子节点
        right = 2*root+2  # 右子节点
        if left < node_count and arr[left] > arr[cur_max]:
            cur_max = left
        if right < node_count and arr[right] > arr[cur_max]:
            cur_max = right
        if root != cur_max:  # 如果最大值位置变更
            arr[root], arr[cur_max] = arr[cur_max], arr[root]
            heapify(arr, node_count, cur_max)

    def build_max_heap(arr):
        n = len(arr)
        for i in range(n // 2 - 1, -1, -1):  # 建立大顶堆。必须反向构建【小顶堆也是反向构建】。从最后一个非叶子节点【由节点个数和非叶子的左右子节点可得最后一个非叶子节点的索引为l//2-1】开始往根节点方式构建
            heapify(arr, n, i)

    build_max_heap(arrs)
    for i in range(arr_len-1, 0, -1):
        arrs[i], arrs[0] = arrs[0], arrs[i]
        heapify(arrs, i, 0)


if __name__ == '__main__':
    arrs = [2, 6, 1, 7, 13, 4, 60, 45]
    print(f"排序前：{arrs}")
    # mao_pao(arrs)
    # cha_ru(arrs)
    # xuan_ze(arrs)
    # kuai_pai(arrs)
    # gui_bing(arrs)
    dui_pai(arrs)
    print(f"排序后：{arrs}")
