#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2026/3/3 10:29
# @Author  : gaohuan
# @Email   : 
# @FileName: self_crawler.py
# @Desc    :
import requests
from bs4 import BeautifulSoup

ROOT_URL = "https://deepwiki.com"
MARK_URL_PREFIX = "/modelscope/agentscope"
SAVE_DIR = "/modelscope/agentscope"


def create_save_dir():
    """创建保存HTML文件的目录（不存在则创建）"""
    import os
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"📁 创建保存目录：{SAVE_DIR}")


if __name__ == '__main__':
    url = "https://deepwiki.com/modelscope/agentscope"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    resp = requests.get(url, headers=headers)
    soup = BeautifulSoup(resp.text, "html")
    print("页面标题：", soup.title.string)
    for a in soup.find_all("a"):
        if a.get("href").startswith(MARK_URL_PREFIX):
            print(f"{a.string} -> {a.get('href')}")
