def gui_bing(arr):
    l = len(arr)
    if l > 1:
        mid = l // 2  # 拆分标记
        left = arr[:mid]  # 左子组
        right = arr[mid:]  # 右子组
        # 拆分左右子组可以调换先后，不影响
        gui_bing(right)  # 递归处理右子组
        gui_bing(left)  # 递归处理左子组
        # 注意下述处理的arr left right都是递归逻辑中当次的数组
        # 初始化各数组的处理标识索引为0，也即都从左边第一个元素开始比较合并
        print('执行合并操作')
        k = 0  # arr索引
        i = 0  # 左子组索引
        j = 0  # 右子组索引
        # left right尚未处理完毕时，比较合并
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                arr[k] = left[i]
                i = i + 1
            else:
                arr[k] = right[j]
                j = j + 1
            k = k + 1
        # right已处理完毕，left未处理完毕，直接收集left
        while i < len(left):
            arr[k] = left[i]
            i = i + 1
            k = k + 1

        # left已处理完毕，right未处理完毕，直接收集right
        while j < len(right):
            arr[k] = right[j]
            j = j + 1
            k = k + 1


if __name__ == '__main__':
    arr = [5, 3, 9, 10, 4, 7]
    print(arr)
    gui_bing(arr)
    print(arr)