#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2026/1/22 17:01
# @Author  : gaohuan
# @Email   : 
# @FileName: all_sort.py
# @Desc    :


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
    if i < len(left):
        arr[k:] = left[i:]
    if j < len(right):
        arr[k:] = right[j:]


def dui_pai(arr: list):
    n = len(arr)
    if n < 2:
        return

    def heapify(node_count, root):
        cur_max = root
        left = 2*root+1
        right = 2*root+2
        if left < node_count and arr[left] > arr[cur_max]:
            cur_max = left
        if right < node_count and arr[right] > arr[cur_max]:
            cur_max = right
        if cur_max != root:
            arr[root], arr[cur_max] = arr[cur_max], arr[root]
            heapify(node_count, cur_max)

    for i in range(n//2 - 1, -1, -1):
        heapify(n, i)

    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(i, 0)


def kuai_pai(arr: list):
    n = len(arr)
    if n < 2:
        return

    def quick_sort(low, high):
        if low >= high:
            return

        def get_mid(low, high):
            mid_value = arr[high]
            mid_ind = low
            for i in range(low, high):
                if arr[i] < mid_value:
                    arr[i], arr[mid_ind] = arr[mid_ind], arr[i]
                    mid_ind = mid_ind + 1
            arr[mid_ind], arr[high] = arr[high], arr[mid_ind]
            return mid_ind

        mid = get_mid(low, high)
        quick_sort(low, mid-1)
        quick_sort(mid+1, high)

    quick_sort(0, n-1)


def mao_pao(arr: list):
    n = len(arr)
    if n < 2:
        return
    for i in range(n):
        for j in range(n-1-i):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]


def xuan_ze(arr: list):
    n = len(arr)
    if n < 2:
        return
    for i in range(n-1):
        cur_min = i
        for j in range(i+1, n):
            if arr[j] < arr[cur_min]:
                cur_min = j
        arr[i], arr[cur_min] = arr[cur_min], arr[i]


def cha_ru(arr: list):
    n = len(arr)
    if n < 2:
        return
    for i in range(1, n):
        key = arr[i]
        j = i-1
        while j > -1 and arr[j] > key:
            arr[j+1] = arr[j]
            j = j - 1
        arr[j+1] = key


if __name__ == '__main__':
    arr = [5, 4, 3, 16, 32, 9, 1, 61]
    print(f'排序前：{arr}')
    # gui_bing(arr)
    # dui_pai(arr)
    # kuai_pai(arr)
    # mao_pao(arr)
    # xuan_ze(arr)
    cha_ru(arr)
    print(f'排序后：{arr}')
