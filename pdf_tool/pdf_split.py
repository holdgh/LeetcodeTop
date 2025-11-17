#!/usr/bin/env python3
"""
增强版PDF拆分工具
支持自定义命名模式、页范围选择等功能
"""

import os
import re
from PyPDF2 import PdfReader, PdfWriter
from pathlib import Path


class PDFSplitter:
    """PDF拆分器类"""

    def __init__(self):
        self.supported_formats = ['.pdf']

    def validate_pdf_file(self, file_path):
        """验证PDF文件"""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        if path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"不支持的文件格式: {path.suffix}")

        # 尝试打开文件验证是否为有效PDF
        try:
            with open(file_path, 'rb') as f:
                PdfReader(f)
        except Exception as e:
            raise ValueError(f"无效的PDF文件: {str(e)}")

    def split_pdf_advanced(self, input_pdf_path, output_dir, pages_per_split=10,
                           naming_pattern=None, start_page=1, end_page=None):
        """
        增强版PDF拆分

        参数:
            input_pdf_path: 输入PDF路径
            output_dir: 输出目录
            pages_per_split: 每文件页数
            naming_pattern: 命名模式，如"文档_第{}部分"
            start_page: 开始页码（从1开始）
            end_page: 结束页码
        """

        # 验证输入文件
        self.validate_pdf_file(input_pdf_path)

        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 读取PDF
        reader = PdfReader(input_pdf_path)
        total_pages = len(reader.pages)

        # 处理页范围
        start_page = max(1, start_page) - 1  # 转换为0基索引
        if end_page is None:
            end_page = total_pages
        else:
            end_page = min(end_page, total_pages)

        actual_pages = end_page - start_page

        if actual_pages <= 0:
            raise ValueError("无效的页范围")

        # 计算拆分数量
        num_splits = (actual_pages + pages_per_split - 1) // pages_per_split

        print(f"📄 文件: {Path(input_pdf_path).name}")
        print(f"📖 总页数: {total_pages}")
        print(f"🔢 处理页范围: {start_page + 1}-{end_page} (共{actual_pages}页)")
        print(f"✂️  拆分规则: 每 {pages_per_split} 页一个文件")
        print(f"📁 将生成: {num_splits} 个文件")
        print("-" * 50)

        generated_files = []
        original_name = Path(input_pdf_path).stem

        for split_index in range(num_splits):
            # 计算页范围
            current_start = start_page + split_index * pages_per_split
            current_end = min(start_page + (split_index + 1) * pages_per_split, end_page)

            # 创建写入器
            writer = PdfWriter()

            # 添加页面
            for page_num in range(current_start, current_end):
                writer.add_page(reader.pages[page_num])

            # 生成文件名
            if naming_pattern:
                filename = naming_pattern.format(split_index + 1) + ".pdf"
            else:
                filename = f"{original_name}_{split_index + 1:03d}.pdf"

            output_filepath = output_path / filename

            # 写入文件
            with open(output_filepath, 'wb') as out_file:
                writer.write(out_file)

            generated_files.append(str(output_filepath))

            # 显示进度
            page_range_str = f"页 {current_start + 1}-{current_end}"
            file_size = output_filepath.stat().st_size // 1024  # KB
            print(f"✅ 生成: {filename} ({page_range_str}, {file_size}KB)")

        total_size = sum(Path(f).stat().st_size for f in generated_files) // 1024
        print("-" * 50)
        print(f"🎉 拆分完成！共生成 {len(generated_files)} 个文件，总大小: {total_size}KB")

        return generated_files

    def get_pdf_info(self, pdf_path):
        """获取PDF文件信息"""
        self.validate_pdf_file(pdf_path)

        reader = PdfReader(pdf_path)
        info = {
            'pages': len(reader.pages),
            'author': reader.metadata.get('/Author', '未知'),
            'title': reader.metadata.get('/Title', '未知'),
            'subject': reader.metadata.get('/Subject', '未知'),
        }

        return info


def main():
    """命令行主函数"""
    import sys

    if len(sys.argv) < 4:
        print("""
📚 PDF拆分工具 - 增强版

使用方法:
  python pdf_splitter_advanced.py <输入文件> <输出目录> <每文件页数> [选项]

选项:
  --name-pattern "模式"   命名模式，使用{}作为序号占位符
                         示例: "文档_第{}部分"
  --start-page N         开始页码 (默认: 1)
  --end-page N           结束页码 (默认: 到文件末尾)
  --info                 仅显示文件信息，不拆分

示例:
  python pdf_splitter_advanced.py doc.pdf ./output 10
  python pdf_splitter_advanced.py doc.pdf ./output 5 --name-pattern "章节_{}"
  python pdf_splitter_advanced.py doc.pdf ./output 10 --start-page 5 --end-page 50
  python pdf_splitter_advanced.py doc.pdf ./output 10 --info
        """)
        return

    splitter = PDFSplitter()

    try:
        # 解析参数
        input_file = sys.argv[1]
        output_dir = sys.argv[2]
        pages_per_split = int(sys.argv[3])

        # 解析可选参数
        naming_pattern = None
        start_page = 1
        end_page = None
        show_info_only = False

        i = 4
        while i < len(sys.argv):
            if sys.argv[i] == '--name-pattern' and i + 1 < len(sys.argv):
                naming_pattern = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == '--start-page' and i + 1 < len(sys.argv):
                start_page = int(sys.argv[i + 1])
                i += 2
            elif sys.argv[i] == '--end-page' and i + 1 < len(sys.argv):
                end_page = int(sys.argv[i + 1])
                i += 2
            elif sys.argv[i] == '--info':
                show_info_only = True
                i += 1
            else:
                i += 1

        if show_info_only:
            # 显示文件信息
            info = splitter.get_pdf_info(input_file)
            print(f"📄 文件信息: {Path(input_file).name}")
            print(f"📖 页数: {info['pages']}")
            print(f"👤 作者: {info['author']}")
            print(f"📝 标题: {info['title']}")
            print(f"📋 主题: {info['subject']}")
        else:
            # 执行拆分
            result = splitter.split_pdf_advanced(
                input_pdf_path=input_file,
                output_dir=output_dir,
                pages_per_split=pages_per_split,
                naming_pattern=naming_pattern,
                start_page=start_page,
                end_page=end_page
            )

    except Exception as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()