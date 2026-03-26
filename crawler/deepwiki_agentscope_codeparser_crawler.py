#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2026/3/2 16:42
# @Author  : gaohuan
# @Email   : 
# @FileName: deepwiki_agentscope_codeparser_crawler.py
# @Desc    :
import requests
import re
import time
from urllib.parse import urljoin, urlparse
from requests.exceptions import RequestException, HTTPError, ConnectionError, Timeout

# 全局配置（可根据需要调整）
BASE_URL = "https://deepwiki.com/modelscope/agentscope"  # 主目录URL
SAVE_DIR = "deepwiki_agentscope_html"  # 保存所有HTML文件的文件夹
DELAY = 1  # 每次请求间隔（秒），避免频繁请求
RETRY_TIMES = 2  # 单个链接爬取失败时的重试次数
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://deepwiki.com/",
    "Connection": "keep-alive"
}


def create_save_dir():
    """创建保存HTML文件的目录（不存在则创建）"""
    import os
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"📁 创建保存目录：{SAVE_DIR}")


def get_valid_links(main_url: str) -> list:
    """
    爬取主页面，解析出所有属于该目录下的有效内部链接

    Args:
        main_url: 主目录URL

    Returns:
        list: 去重后的有效内部链接列表
    """
    try:
        response = requests.get(main_url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        response.encoding = response.apparent_encoding

        # 正则匹配页面中的<a>标签里的href属性（匹配所有以/modelscope/agentscope开头的链接）
        # 匹配规则：href="xxx" 或 href='xxx'，且xxx以/modelscope/agentscope开头
        pattern = r'href=["\'](/modelscope/agentscope[^"\']+)["\']'
        raw_links = re.findall(pattern, response.text)

        # 补全URL（相对路径转绝对路径）+ 去重 + 过滤无效链接
        valid_links = set()  # 用集合去重
        main_domain = f"{urlparse(main_url).scheme}://{urlparse(main_url).netloc}"

        for link in raw_links:
            full_url = urljoin(main_domain, link)
            # 只保留属于deepwiki.com且路径包含/modelscope/agentscope的链接
            if urlparse(full_url).netloc == urlparse(main_url).netloc and "/modelscope/agentscope" in full_url:
                valid_links.add(full_url)

        # 把主页面本身也加入列表（避免遗漏）
        valid_links.add(main_url)

        # 转成列表并排序
        link_list = sorted(list(valid_links))
        print(f"🔍 解析出有效链接数量：{len(link_list)}")
        for i, link in enumerate(link_list, 1):
            print(f"   {i}. {link}")

        return link_list

    except Exception as e:
        print(f"❌ 解析链接失败：{e}")
        return []


def crawl_single_page(url: str, retry: int = RETRY_TIMES) -> str:
    """
    爬取单个URL的HTML内容（带重试机制）

    Args:
        url: 目标URL
        retry: 剩余重试次数

    Returns:
        str: 爬取到的HTML内容，失败则返回空字符串
    """
    if retry <= 0:
        print(f"⚠️ {url} 重试次数用尽，爬取失败")
        return ""

    try:
        print(f"🚀 正在爬取：{url} (剩余重试次数：{retry})")
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        time.sleep(DELAY)  # 请求间隔，遵守爬虫礼仪
        return response.text

    except (HTTPError, ConnectionError, Timeout) as e:
        print(f"⚠️ {url} 爬取失败：{e}，{retry - 1}秒后重试...")
        time.sleep(1)  # 重试前等待1秒
        return crawl_single_page(url, retry - 1)
    except RequestException as e:
        print(f"⚠️ {url} 请求异常：{e}")
        return ""


def save_html_to_file(html_content: str, url: str):
    """
    将HTML内容保存到文件（根据URL生成合法的文件名）

    Args:
        html_content: HTML内容
        url: 对应的URL
    """
    if not html_content:
        return

    # 生成合法文件名：替换URL中的特殊字符，避免文件名非法
    # 示例：https://deepwiki.com/modelscope/agentscope/xxx → agentscope_xxx.html
    url_path = urlparse(url).path
    # 去掉开头的/modelscope/agentscope，替换特殊字符为下划线
    file_name = url_path.replace("/modelscope/agentscope", "").strip("/")
    # 如果是主页面，命名为index
    if not file_name:
        file_name = "index"
    # 替换非法字符（\ / : * ? " < > |）为下划线
    file_name = re.sub(r'[\\/:*?"<>|]', "_", file_name)
    # 限制文件名长度，避免超出系统限制
    file_name = file_name[:50] if len(file_name) > 50 else file_name
    # 拼接完整保存路径
    save_path = f"{SAVE_DIR}/{file_name}.html"

    try:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"✅ 保存成功：{save_path} (大小：{len(html_content)} 字符)")
    except Exception as e:
        print(f"❌ 保存失败 {save_path}：{e}")


def batch_crawl():
    """批量爬取所有解析出的链接"""
    # 1. 创建保存目录
    create_save_dir()
    # 2. 解析主页面的所有有效链接
    links = get_valid_links(BASE_URL)
    if not links:
        print("❌ 未解析到任何有效链接，爬取终止")
        return
    # 3. 批量爬取并保存
    success_count = 0
    fail_count = 0
    print("\n============ 开始批量爬取 ============")
    for link in links:
        html_content = crawl_single_page(link)
        if html_content:
            save_html_to_file(html_content, link)
            success_count += 1
        else:
            fail_count += 1
    # 4. 输出爬取统计
    print("\n============ 爬取完成 ============")
    print(f"📊 统计结果：")
    print(f"   总链接数：{len(links)}")
    print(f"   成功数：{success_count}")
    print(f"   失败数：{fail_count}")
    print(f"   保存目录：{SAVE_DIR}")


if __name__ == "__main__":
    # 执行批量爬取
    batch_crawl()