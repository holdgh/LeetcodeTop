import jieba
import jieba.analyse

if __name__ == '__main__':
    # 1. 加载 jieba 内置停用词表（也可替换为自定义停用词表）
    jieba.analyse.set_stop_words("stopwords.txt")  # 若无自定义表，可省略，使用内置默认表

    # 2. 分词并过滤停用词（extract_tags 会自动剔除停用词，返回核心关键词）
    text = "电机碳刷维护的标准流程是什么？"
    # 分词（含停用词）
    words_with_stop = jieba.lcut(text)
    print("分词结果（含停用词）：", words_with_stop)  # ['电机', '碳刷维护', '的', '标准流程', '是', '什么', '？']

    # 提取核心关键词（自动过滤停用词）
    keywords = jieba.analyse.extract_tags(text, topK=10, withWeight=False)
    print("过滤后核心关键词：", keywords)  # ['碳刷维护', '标准流程', '电机']
