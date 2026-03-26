#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2026/1/23 11:26
# @Author  : gaohuan
# @Email   : 
# @FileName: baidu.py
# @Desc    :
from modelscope.hub.snapshot_download import snapshot_download


if __name__ == '__main__':
    # 下载ERNIE-tiny模型（国内源，稳定）
    model_dir = snapshot_download(
        "nghuyong/ernie-tiny",
        cache_dir="./ernie-tiny-local",  # 本地保存路径
        revision="master"
    )
    print(f"模型已下载到：{model_dir}")