
def quick_sort(nums):

    def get_mid(nums, low, high):  # 取high处值为临界值【目标值】，将其与当前范围的其他元素，逐个比较，小于目标值的移动到其左侧，剩余大于目标值的自动位于其右侧，返回目标值最终的正确位置
        target_value = nums[high]
        mid = low
        for i in range(low, high):
            if target_value >= nums[i]:
                nums[i], nums[mid] = nums[mid], nums[i]  # 不大于目标值的元素左移，保证左侧皆不大于目标值，右侧皆大于目标值
                mid = mid + 1  # 目标值位置右移一位
        nums[mid], nums[high] = nums[high], nums[mid]
        return mid

    def quick(nums, low, high):
        if low >= high:  # 递归终止条件，只有一个元素自然有序且满足快排思想
            return
        mid = get_mid(nums, low, high)
        quick(nums, low, mid - 1)
        quick(nums, mid + 1, high)

    l = len(nums)
    if l < 2:
        return
    quick(nums, 0, l- 1)


if __name__ == '__main__':
    nums = [3, 7, 5, 1, 9, 10, 6, 4, 12, 30, 22, 11]
    print(f'排序前：{nums}')
    quick_sort(nums)
    print(f'排序后：{nums}')
