import jieba
import re
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import SentenceTransformer, util  # 向量检索依赖


# 1. 原有BM25检索类（无需修改得分逻辑，保留原始排名即可）
class BM25KeywordRetriever:
    """BM25 关键词检索器（保留原始得分和排名）"""

    def __init__(self):
        self.corpus = []
        self.tokenized_corpus = []
        self.bm25_model = None

    def text_cleaning(self, text):
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：""''（）《》【】]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def chinese_tokenize(self, text):
        stop_words = {"的", "地", "得", "了", "着", "过", "啊", "呀", "吗", "呢", "吧",
                      "之", "乎", "者", "也", "而", "且", "则", "与", "或", "若", "因",
                      "为", "以", "于", "对", "和", "在", "是", "我", "你", "他", "我们"}
        words = [word for word in jieba.lcut(text) if word not in stop_words and word.strip()]
        return words

    def build_index(self, document_list):
        self.corpus = [self.text_cleaning(doc) for doc in document_list]
        self.tokenized_corpus = [self.chinese_tokenize(doc) for doc in self.corpus]
        self.bm25_model = BM25Okapi(self.tokenized_corpus)
        print(f"BM25 索引构建完成！共加载 {len(self.corpus)} 篇文档")

    def retrieve_with_rank(self, query, top_k=10):
        """返回带排名的BM25检索结果（无需归一化得分）"""
        if not self.bm25_model:
            raise ValueError("请先调用 build_index 构建索引！")

        cleaned_query = self.text_cleaning(query)
        tokenized_query = self.chinese_tokenize(cleaned_query)
        scores = self.bm25_model.get_scores(tokenized_query)

        # 按得分降序排序，获取（文档索引、原始得分、排名）
        sorted_indices = scores.argsort()[::-1]
        ranked_results = []
        for rank, idx in enumerate(sorted_indices[:top_k], 1):  # 排名从1开始
            ranked_results.append({
                "document": self.corpus[idx],
                "raw_bm25_score": round(scores[idx], 4),
                "document_index": idx,
                "bm25_rank": rank
            })

        return ranked_results


# 2. 简单向量检索类（返回带排名的结果）
class VectorRetriever:
    """向量检索器（基于Sentence-BERT，返回0-1相似度和排名）"""

    def __init__(self, model_name=r'C:\Users\gaohu\aiModel\text2vec-base-chinese'):
        self.model = SentenceTransformer(model_name)
        self.corpus = []
        self.corpus_embeddings = None

    def build_index(self, document_list):
        self.corpus = [doc.strip() for doc in document_list]
        self.corpus_embeddings = self.model.encode(self.corpus, convert_to_tensor=True)
        print(f"向量索引构建完成！共加载 {len(self.corpus)} 篇文档")

    def retrieve_with_rank(self, query, top_k=10):
        """返回带排名的向量检索结果"""
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        # 计算余弦相似度（0-1区间）
        cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
        cos_scores = cos_scores.cpu().numpy()

        # 按相似度降序排序，获取（文档索引、相似度得分、排名）
        sorted_indices = cos_scores.argsort()[::-1]
        ranked_results = []
        for rank, idx in enumerate(sorted_indices[:top_k], 1):  # 排名从1开始
            ranked_results.append({
                "document": self.corpus[idx],
                "vector_similarity": round(cos_scores[idx], 4),
                "document_index": idx,
                "vector_rank": rank
            })

        return ranked_results


# 3. RRF融合实现（无需归一化，仅用排名）
def reciprocal_rank_fusion(bm25_ranked_results, vector_ranked_results, k=6, top_k=10):
    """
    倒数排名融合（RRF）
    :param bm25_ranked_results: BM25带排名的结果
    :param vector_ranked_results: 向量检索带排名的结果
    :param k: 平滑系数（经验值6-10）
    :param top_k: 融合后返回的文档数量
    :return: 融合排序后的结果
    """
    # 构建文档索引到排名的映射
    doc_rank_map = {}

    # 录入BM25排名
    for res in bm25_ranked_results:
        doc_idx = res["document_index"]
        doc_rank_map[doc_idx] = {
            "document": res["document"],
            "bm25_rank": res["bm25_rank"],
            "vector_rank": None,
            "rrf_score": 0.0
        }

    # 录入向量检索排名
    for res in vector_ranked_results:
        doc_idx = res["document_index"]
        if doc_idx in doc_rank_map:
            doc_rank_map[doc_idx]["vector_rank"] = res["vector_rank"]
        else:
            doc_rank_map[doc_idx] = {
                "document": res["document"],
                "bm25_rank": None,
                "vector_rank": res["vector_rank"],
                "rrf_score": 0.0
            }

    # 计算RRF得分
    for doc_idx in doc_rank_map:
        rrf_sum = 0.0
        # BM25排名贡献（若存在）
        if doc_rank_map[doc_idx]["bm25_rank"] is not None:
            rrf_sum += 1 / (k + doc_rank_map[doc_idx]["bm25_rank"])
        # 向量排名贡献（若存在）
        if doc_rank_map[doc_idx]["vector_rank"] is not None:
            rrf_sum += 1 / (k + doc_rank_map[doc_idx]["vector_rank"])
        doc_rank_map[doc_idx]["rrf_score"] = round(rrf_sum, 4)

    # 按RRF得分降序排序，返回Top-K
    sorted_hybrid_results = sorted(
        doc_rank_map.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )[:top_k]

    return sorted_hybrid_results


# 4. 测试RRF融合效果（无归一化，直接用排名）
if __name__ == "__main__":
    # 准备测试文档
    test_documents = [
        "RAG（检索增强生成）的核心组件包括知识库、向量数据库、大语言模型，BM25 是常用的关键词检索算法。",
        "BM25 算法是 TF-IDF 的改进版，引入了文档长度归一化和词频饱和机制，适合全文检索场景。",
        "大语言模型容易产生幻觉，通过 RAG 检索外部知识库可以有效减少虚假信息生成。",
        "Windows 系统下使用 Python 实现 BM25 检索，需要安装 jieba 和 rank_bm25 库。",
        "向量检索注重语义相似性，BM25 注重关键词匹配，两者结合可实现 Hybrid RAG 提升效果。",
        "中文文本检索需要先进行分词处理，jieba 是 Python 中常用的中文分词工具。"
    ]

    # 初始化并构建索引
    bm25_retriever = BM25KeywordRetriever()
    bm25_retriever.build_index(test_documents)

    vector_retriever = VectorRetriever()
    vector_retriever.build_index(test_documents)

    # 用户查询
    user_query = "Windows 下 Python 实现 BM25 检索"

    # 分别获取带排名的检索结果
    bm25_ranked_results = bm25_retriever.retrieve_with_rank(user_query, top_k=5)
    vector_ranked_results = vector_retriever.retrieve_with_rank(user_query, top_k=5)

    # RRF融合（无需任何归一化）
    hybrid_results = reciprocal_rank_fusion(bm25_ranked_results, vector_ranked_results, k=6, top_k=5)

    # 输出结果
    print(f"查询：{user_query}")
    print("-" * 80)
    print("BM25检索排名结果：")
    for res in bm25_ranked_results[:2]:
        print(f"排名{res['bm25_rank']} | 原始得分{res['raw_bm25_score']} | 文档：{res['document'][:50]}...")

    print("\n向量检索排名结果：")
    for res in vector_ranked_results[:2]:
        print(f"排名{res['vector_rank']} | 相似度{res['vector_similarity']} | 文档：{res['document'][:50]}...")

    print("\nRRF融合后排名结果（无需归一化）：")
    for i, res in enumerate(hybrid_results[:2], 1):
        print(f"融合排名{i} | RRF得分{res['rrf_score']} | 文档：{res['document'][:50]}...")
        print(f"  - BM25排名：{res['bm25_rank']} | 向量排名：{res['vector_rank']}")
    print("-" * 80)