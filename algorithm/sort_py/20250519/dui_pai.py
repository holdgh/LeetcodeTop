def dui_pai(arr):
    l = len(arr)
    if l < 2:
        return
    # 根据左右子节点与其父节点的索引关系式，结合数组索引最大长度限制，可以得出最后一个非叶子节点的索引为l//2-1
    # 构造大顶堆
    for i in range(l//2-1, -1, -1):  # 反向调整【这是必须的，以满足大顶堆的特点--根节点是最大值；任一父节点大于其左右子节点】【构建小顶堆，也需要反向调整】
        create_max_dui_for_index(arr, l, i)
    # 沿用“冒泡排序”思路【每次取堆顶【最大值】，并将其与未排序的最后一个元素置换】
    for j in range(l-1, 0, -1):
        arr[j], arr[0] = arr[0], arr[j]  # 将当前最大值置后
        create_max_dui_for_index(arr, j, 0)  # 这里构建大顶堆的目的是快速找到当前剩余元素【未排序元素，注意j的变化与函数create_max_dui_for_index中对n的定义【限制子节点的索引范围】】的最大值


def create_max_dui_for_index(arr, n, i):
    cur_max = i
    left, right = 2 * i + 1, 2 * i + 2
    if left < n and arr[i] < arr[left]:
        cur_max = left
    if right < n and arr[i] < arr[right]:
        cur_max = right
    if cur_max != i:
        arr[i], arr[cur_max] = arr[cur_max], arr[i]  # 将子节点的较大值赋予父节点【没有在上面直接置换，这里可以减少一次置换操作【当父节点小于左右子节点时，上述就需要置换两次】】
        create_max_dui_for_index(arr, n, cur_max)


if __name__ == '__main__':
    arr = [3, 2, 5, 4, 7, 6, 10, 1]
    print(f'排序前：{arr}')
    dui_pai(arr)
    print(f'排序后：{arr}')