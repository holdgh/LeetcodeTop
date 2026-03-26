#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2026/3/2 16:49
# @Author  : gaohuan
# @Email   : 
# @FileName: deepwiki_agentscope_codeparser_all_crawler.py
# @Desc    :
import requests
import re
import os
import time
from urllib.parse import urljoin, urlparse, unquote
from requests.exceptions import RequestException, HTTPError, ConnectionError, Timeout

# 全局配置
BASE_URL = "https://deepwiki.com/modelscope/agentscope"
SAVE_ROOT_DIR = "deepwiki_agentscope_complete"  # 本地保存根目录
DELAY = 1.5  # 请求间隔（秒）
RETRY_TIMES = 2  # 重试次数
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": BASE_URL,
    "Connection": "keep-alive"
}

# 需要爬取的资源类型
RESOURCE_EXTENSIONS = {
    # HTML文件
    'html': ['', '.html', '.htm', '.md'],
    # 样式文件
    'css': ['.css'],
    # 脚本文件
    'js': ['.js'],
    # 图片文件
    'image': ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.ico'],
    # 字体文件
    'font': ['.woff', '.woff2', '.ttf', '.eot', '.otf'],
    # 其他资源
    'other': ['.json', '.xml', '.txt']
}

# 已爬取的URL集合（避免重复爬取）
crawled_urls = set()
# 待爬取的URL队列
url_queue = []


def init_dir_structure():
    """初始化本地目录结构"""
    # 创建根目录
    if not os.path.exists(SAVE_ROOT_DIR):
        os.makedirs(SAVE_ROOT_DIR)
        print(f"📁 创建根目录：{SAVE_ROOT_DIR}")

    # 创建资源子目录（分类存放，也可按原路径）
    resource_dirs = ['css', 'js', 'images', 'fonts', 'html', 'others']
    for dir_name in resource_dirs:
        dir_path = os.path.join(SAVE_ROOT_DIR, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


def get_file_extension(url: str) -> str:
    """获取URL的文件扩展名"""
    parsed_url = urlparse(url)
    path = parsed_url.path
    ext = os.path.splitext(path)[1].lower()
    return ext


def get_resource_type(ext: str) -> str:
    """根据扩展名判断资源类型"""
    for res_type, exts in RESOURCE_EXTENSIONS.items():
        if ext in exts:
            return res_type
    return 'other'


def get_local_save_path(url: str, base_domain: str) -> str:
    """
    根据URL生成本地保存路径
    保持和线上一致的目录结构
    """
    parsed_url = urlparse(url)
    path = parsed_url.path
    query = parsed_url.query

    # 移除base_domain，只保留路径部分
    full_path = path
    if query:
        full_path += f"?{query}"

    # 解码URL中的特殊字符
    full_path = unquote(full_path)

    # 处理根路径（主页面）
    if full_path == '/' or full_path == '':
        local_path = os.path.join(SAVE_ROOT_DIR, "index.html")
        return local_path

    # 拼接本地路径
    local_path = os.path.join(SAVE_ROOT_DIR, full_path.lstrip('/'))

    # 处理无扩展名的HTML页面（如 /modelscope/agentscope → modelscope/agentscope.html）
    ext = get_file_extension(url)
    if not ext and 'html' in get_resource_type(ext):
        if not local_path.endswith('.html'):
            local_path += '.html'

    # 创建父目录（如果不存在）
    parent_dir = os.path.dirname(local_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    return local_path


def replace_absolute_paths(content: str, base_url: str, local_path: str) -> str:
    """
    将内容中的绝对路径替换为本地相对路径
    保证本地打开时资源能正常加载
    """
    parsed_base = urlparse(base_url)
    base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"

    # 匹配所有绝对URL
    patterns = [
        # 匹配src="https://xxx" 或 href="https://xxx"
        r'(src|href|url)\s*=\s*["\'](' + re.escape(base_domain) + r'[^"\']+)["\']',
        # 匹配url(https://xxx) 样式中的URL
        r'url\(\s*["\']?(' + re.escape(base_domain) + r'[^"\')]+)["\']?\s*\)'
    ]

    def replace_match(match):
        """替换单个匹配项"""
        try:
            # 获取匹配到的完整URL
            if match.group(1).lower() == 'url':
                full_url = match.group(1)
                resource_url = match.group(2)
            else:
                attr = match.group(1)
                resource_url = match.group(2)

            # 生成资源的本地路径
            resource_local_path = get_local_save_path(resource_url, base_domain)
            # 计算相对路径
            current_dir = os.path.dirname(local_path)
            rel_path = os.path.relpath(resource_local_path, current_dir)
            # Windows路径转URL路径（/ 而非 \）
            rel_path = rel_path.replace('\\', '/')

            # 返回替换后的内容
            if match.group(1).lower() == 'url':
                return f'url("{rel_path}")'
            else:
                return f'{attr}="{rel_path}"'
        except:
            return match.group(0)

    # 执行替换
    for pattern in patterns:
        content = re.sub(pattern, replace_match, content, flags=re.IGNORECASE)

    return content


def crawl_resource(url: str, retry: int = RETRY_TIMES) -> bytes | str | None:
    """
    爬取单个资源（支持文本和二进制资源）
    返回：文本内容（HTML/CSS/JS）或二进制内容（图片/字体），失败返回None
    """
    if url in crawled_urls:
        return None

    if retry <= 0:
        print(f"⚠️ {url} 重试次数用尽，爬取失败")
        return None

    try:
        print(f"🚀 正在爬取：{url} (剩余重试：{retry})")
        response = requests.get(url, headers=HEADERS, timeout=20, stream=True)
        response.raise_for_status()

        # 标记为已爬取
        crawled_urls.add(url)
        time.sleep(DELAY)

        # 判断资源类型，决定返回文本还是二进制
        ext = get_file_extension(url)
        res_type = get_resource_type(ext)

        if res_type in ['html', 'css', 'js', 'other']:
            # 文本资源，处理编码
            response.encoding = response.apparent_encoding
            return response.text
        else:
            # 二进制资源（图片/字体）
            return response.content

    except (HTTPError, ConnectionError, Timeout) as e:
        print(f"⚠️ {url} 爬取失败：{e}，{retry - 1}秒后重试...")
        time.sleep(1)
        return crawl_resource(url, retry - 1)
    except RequestException as e:
        print(f"⚠️ {url} 请求异常：{e}")
        return None
    except Exception as e:
        print(f"⚠️ {url} 未知错误：{e}")
        return None


def save_resource(content, local_path: str, is_binary: bool = False):
    """保存资源到本地文件"""
    try:
        # 打开模式：二进制用wb，文本用w
        mode = 'wb' if is_binary else 'w'
        encoding = None if is_binary else 'utf-8'

        with open(local_path, mode, encoding=encoding) as f:
            f.write(content)

        # 计算文件大小
        size = len(content)
        size_str = f"{size / 1024:.2f} KB" if size > 1024 else f"{size} B"
        print(f"✅ 保存成功：{local_path} ({size_str})")
        return True
    except Exception as e:
        print(f"❌ 保存失败 {local_path}：{e}")
        return False


def extract_resources_from_html(html_content: str, base_url: str) -> list:
    """从HTML内容中提取所有需要爬取的资源URL"""
    parsed_base = urlparse(base_url)
    base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"

    # 匹配所有资源URL的正则
    resource_patterns = [
        # 匹配src/href属性
        r'(src|href)\s*=\s*["\']([^"\']+)["\']',
        # 匹配样式中的url()
        r'url\(\s*["\']?([^"\')]+)["\']?\s*\)'
    ]

    resource_urls = set()

    for pattern in resource_patterns:
        matches = re.findall(pattern, html_content, flags=re.IGNORECASE)
        for match in matches:
            try:
                # 处理匹配结果
                if len(match) == 2:
                    # src/href 匹配
                    resource_url = match[1]
                else:
                    # url() 匹配
                    resource_url = match[0]

                # 跳过空值、锚点、mailto、tel等
                if not resource_url or resource_url.startswith(('#', 'mailto:', 'tel:', 'javascript:')):
                    continue

                # 补全相对路径为绝对URL
                full_url = urljoin(base_domain, resource_url)

                # 只保留当前域名的资源
                if urlparse(full_url).netloc == parsed_base.netloc:
                    resource_urls.add(full_url)
            except:
                continue

    # 同时提取页面中的内部链接（其他HTML页面）
    # 匹配<a>标签中指向/modelscope/agentscope的链接
    link_pattern = r'href=["\'](/modelscope/agentscope[^"\']+)["\']'
    internal_links = re.findall(link_pattern, html_content)
    for link in internal_links:
        full_url = urljoin(base_domain, link)
        if urlparse(full_url).netloc == parsed_base.netloc:
            resource_urls.add(full_url)
            # 添加到待爬取队列
            if full_url not in crawled_urls and full_url not in url_queue:
                url_queue.append(full_url)

    return list(resource_urls)


def process_url(url: str):
    """处理单个URL：爬取→提取资源→保存→递归处理"""
    # 1. 爬取当前URL的内容
    content = crawl_resource(url)
    if content is None:
        return

    # 2. 获取本地保存路径
    parsed_base = urlparse(BASE_URL)
    base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"
    local_path = get_local_save_path(url, base_domain)

    # 3. 判断资源类型
    ext = get_file_extension(url)
    res_type = get_resource_type(ext)

    # 4. 处理内容（HTML需要替换路径，提取子资源）
    if res_type == 'html':
        # 替换绝对路径为本地相对路径
        processed_content = replace_absolute_paths(content, base_domain, local_path)
        # 保存处理后的HTML
        save_resource(processed_content, local_path)
        # 提取页面中的所有资源URL
        resource_urls = extract_resources_from_html(processed_content, base_domain)
        # 将新发现的URL加入队列
        for res_url in resource_urls:
            if res_url not in crawled_urls and res_url not in url_queue:
                url_queue.append(res_url)
    else:
        # 非HTML资源，判断是否为二进制
        is_binary = res_type in ['image', 'font']
        # 直接保存
        save_resource(content, local_path, is_binary)


def batch_crawl_complete_website():
    """批量爬取完整网站"""
    # 初始化
    init_dir_structure()
    global crawled_urls, url_queue
    crawled_urls = set()
    url_queue = [BASE_URL]

    print("============ 开始爬取完整网站 ============")
    print(f"📌 起始URL：{BASE_URL}")
    print(f"📂 保存目录：{SAVE_ROOT_DIR}")
    print("----------------------------------------")

    # 处理队列中的URL
    processed_count = 0
    fail_count = 0

    while url_queue:
        # 取出队列第一个URL
        current_url = url_queue.pop(0)
        try:
            process_url(current_url)
            processed_count += 1
        except Exception as e:
            print(f"❌ 处理URL失败 {current_url}：{e}")
            fail_count += 1

        # 打印进度
        print(f"\n📊 进度：已处理 {processed_count} | 待处理 {len(url_queue)} | 失败 {fail_count}")
        print("----------------------------------------")

    # 统计结果
    print("============ 爬取完成 ============")
    print(f"📈 最终统计：")
    print(f"   总处理URL数：{processed_count}")
    print(f"   失败数：{fail_count}")
    print(f"   成功爬取资源数：{len(crawled_urls)}")
    print(f"   📂 所有文件已保存至：{os.path.abspath(SAVE_ROOT_DIR)}")
    print("\n💡 提示：直接打开保存目录中的 index.html 即可在本地浏览完整页面")


if __name__ == "__main__":
    # 执行完整爬取
    batch_crawl_complete_website()