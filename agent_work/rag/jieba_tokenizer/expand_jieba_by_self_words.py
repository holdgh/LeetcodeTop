import jieba


# ===================== 验证分词效果 =====================
if __name__ == "__main__":
    # 测试自定义词库是否生效
    test_text = "RAG 检索优化的核心是语义chunking和三级检索，粗排用BM25稀疏向量检索"
    print("默认分词（未加载词库）：", "/".join(jieba.cut(test_text, cut_all=False)))

    # 重新加载词库（确保生效）
    jieba.load_userdict("rag_custom_dict.txt")
    print("自定义词库分词：", "/".join(jieba.cut(test_text, cut_all=False)))