#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2026/1/29 14:01
# @Author  : gaohuan
# @Email   : 
# @FileName: gsm8k_use_demo.py
# @Desc    :

from modelscope.msdatasets import MsDataset


if __name__ == '__main__':
    ds = MsDataset.load('modelscope/gsm8k', subset_name='main', split='train', trust_remote_code=True)
    print()
