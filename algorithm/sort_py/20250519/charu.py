def cha_ru(arr):
    l = len(arr)
    if l < 2:
        return
    for i in range(1, l):  # 未排序部分，逐个处理，在已排序部分中寻找合适的位置插入
        target = i  # 初始化目标位置
        target_value = arr[i]  # 暂存当前未排序元素的值
        for j in range(i):  # 已排序部分
            if target_value < arr[j]:  # 当已排序的当前元素小于当前未排序元素时，将当前已排序元素移动到临时目标位置，并往前同步更新临时目标位置
                arr[target] = arr[j]
                target = target - 1
            else:  # 说明找到了合适的目标位置
                break
        arr[target] = target_value  # 将当前为排序元素放置到合适的位置


if __name__ == '__main__':
    arr = [3, 2, 5, 4, 7, 6, 9, 8]
    print(f'排序前：{arr}')
    cha_ru(arr)
    print(f'排序后：{arr}')