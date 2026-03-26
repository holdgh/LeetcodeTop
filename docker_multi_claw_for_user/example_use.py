#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2026/3/25 9:55
# @Author  : gaohuan
# @Email   : 
# @FileName: example_use.py
# @Desc    :
# examples/basic_usage.py
import requests

# API基础URL
BASE_URL = "http://localhost:8080/admin"


def register_user(user_id: str, user_name: str):
    """注册新用户"""
    response = requests.post(f"{BASE_URL}/users/{user_id}/register", params={"user_name": user_name})
    return response.json()


def login_user(user_id: str):
    """用户登录"""
    response = requests.post(f"{BASE_URL}/users/{user_id}/login")
    return response.json()


def list_instances():
    """列出所有实例"""
    response = requests.get(f"{BASE_URL}/instances")
    return response.json()


# 使用示例
if __name__ == "__main__":
    # 注册用户
    result = register_user("user456", "李四")
    print("注册结果:", result)

    # 用户登录
    login_result = login_user("user456")
    print("登录结果:", login_result)

    # 列出实例
    instances = list_instances()
    print("实例列表:", instances)