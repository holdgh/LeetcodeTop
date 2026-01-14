import stopwordsiso as stopwords
import jieba


if __name__ == '__main__':
    # 1. 获取中文停用词集合
    stop_words = stopwords.stopwords("zh")
    # 补充领域无关词（如 RAG 场景的通用低价值词）
    stop_words.update({"方法", "流程", "问题", "介绍", "说明"})  # 扩展自定义无关词

    # 2. 分词 + 过滤
    text = "碳刷维护的方法和电机检修的流程介绍"
    jieba.load_userdict("rag_custom_dict.txt")  # 加载自定义词库，优先使用自定义词库
    words = jieba.lcut(text)
    # 过滤逻辑：剔除停用词 + 长度<2的无意义词 + 特殊字符
    filtered_words = [
        word for word in words
        if word not in stop_words  # 剔除停用词
        and len(word) >= 2         # 剔除单字（如“的”已被过滤，还可剔除“和”“或”等）
        and word.strip().isalnum() # 剔除特殊字符（如？、@）
    ]

    print("过滤后关键词：", filtered_words)  # ['碳刷维护', '电机检修']