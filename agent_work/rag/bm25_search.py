import jieba
import re
from rank_bm25 import BM25Okapi  # 主流 BM25 实现（Okapi BM25）


class BM25KeywordRetriever:
    """BM25 关键词检索器（支持中文文本）"""

    def __init__(self):
        self.corpus = []  # 原始文档列表
        self.tokenized_corpus = []  # 分词后的文档列表
        self.bm25_model = None  # BM25 模型实例

    def text_cleaning(self, text):
        """
        文本清洗：去除特殊字符、多余空格、换行符，统一格式
        :param text: 原始文本
        :return: 清洗后的纯文本
        """
        # 去除特殊字符（保留中文、英文、数字、中文标点）
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：""''（）《》【】]', '', text)
        # 去除多余空格和换行符
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def chinese_tokenize(self, text):
        """
        中文分词：使用 jieba 分词，去除停用词（简单停用词表）
        :param text: 清洗后的文本
        :return: 分词后的词汇列表
        """
        # 简单中文停用词表（可根据需求扩展）
        stop_words = {"的", "地", "得", "了", "着", "过", "啊", "呀", "吗", "呢", "吧",
                      "之", "乎", "者", "也", "而", "且", "则", "与", "或", "若", "因",
                      "为", "以", "于", "对", "和", "在", "是", "我", "你", "他", "我们"}

        # 分词并过滤停用词、空字符串
        words = [word for word in jieba.lcut(text) if word not in stop_words and word.strip()]
        return words

    def build_index(self, document_list):
        """
        构建 BM25 索引
        :param document_list: 原始文档列表（每个元素是一篇文档/文本块）
        """
        # 步骤1：批量清洗文本
        self.corpus = [self.text_cleaning(doc) for doc in document_list]

        # 步骤2：批量分词
        self.tokenized_corpus = [self.chinese_tokenize(doc) for doc in self.corpus]

        # 步骤3：初始化 BM25 模型（构建索引）
        self.bm25_model = BM25Okapi(self.tokenized_corpus)
        print(f"BM25 索引构建完成！共加载 {len(self.corpus)} 篇文档")

    def retrieve(self, query, top_k=5):
        """
        执行 BM25 检索，返回 Top-K 相关文档
        :param query: 用户查询语句（中文/英文）
        :param top_k: 返回最相关的前 k 篇文档
        :return: 检索结果列表（包含文档内容、相关度得分）
        """
        if not self.bm25_model:
            raise ValueError("请先调用 build_index 构建索引！")

        # 步骤1：清洗并分词查询语句
        cleaned_query = self.text_cleaning(query)
        tokenized_query = self.chinese_tokenize(cleaned_query)

        # 步骤2：计算查询与所有文档的相关度得分
        scores = self.bm25_model.get_scores(tokenized_query)

        # 步骤3：按得分降序排序，获取 Top-K 文档索引
        top_indices = scores.argsort()[::-1][:top_k]

        # 步骤4：整理检索结果
        results = []
        for idx in top_indices:
            results.append({
                "document": self.corpus[idx],  # 原始文档内容
                "similarity_score": round(scores[idx], 4),  # 相关度得分（保留4位小数）
                "document_index": idx  # 文档在原始列表中的索引
            })

        return results


# -------------------------- 测试使用示例 --------------------------
if __name__ == "__main__":
    # 1. 准备测试文档（模拟 RAG 知识库文本块）
    test_documents = [
        "RAG（检索增强生成）的核心组件包括知识库、向量数据库、大语言模型，BM25 是常用的关键词检索算法。",
        "BM25 算法是 TF-IDF 的改进版，引入了文档长度归一化和词频饱和机制，适合全文检索场景。",
        "大语言模型容易产生幻觉，通过 RAG 检索外部知识库可以有效减少虚假信息生成。",
        "Windows 系统下使用 Python 实现 BM25 检索，需要安装 jieba 和 rank_bm25 库。",
        "向量检索注重语义相似性，BM25 注重关键词匹配，两者结合可实现 Hybrid RAG 提升效果。",
        "中文文本检索需要先进行分词处理，jieba 是 Python 中常用的中文分词工具。"
    ]

    # 2. 初始化并构建 BM25 索引
    bm25_retriever = BM25KeywordRetriever()
    bm25_retriever.build_index(test_documents)

    # 3. 执行查询（模拟用户检索需求）
    user_queries = [
        "BM25 算法的优势",
        "RAG 核心组件有哪些",
        "Windows 下 Python 实现 BM25",
        "如何减少大语言模型幻觉"
    ]

    # 4. 遍历查询并输出结果
    for i, query in enumerate(user_queries, 1):
        print(f"\n========== 第 {i} 个查询：{query} ==========")
        results = bm25_retriever.retrieve(query, top_k=2)  # 返回前2篇最相关文档
        for j, res in enumerate(results, 1):
            print(f"排名 {j}（得分：{res['similarity_score']}）：")
            print(f"文档内容：{res['document']}")
            print(f"文档索引：{res['document_index']}")
            print("-" * 50)