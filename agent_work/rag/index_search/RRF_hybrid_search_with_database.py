# -------------------------- 依赖导入 --------------------------
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
import hashlib
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID, KEYWORD
from whoosh.analysis import Tokenizer, Token, Analyzer
from whoosh.qparser import QueryParser
from whoosh import scoring
import jieba


# -------------------------- FAISS向量检索器（优化版，支持索引复用） --------------------------
class FAISSVectorRetriever:
    """FAISS向量检索器（支持索引复用、文本向量化、Top-K检索）"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        self.index = None
        self.corpus = []
        self.vector_dim = None
        self.index_dir = None
        self.index_file = "vector_index.faiss"
        self.corpus_file = "corpus.txt"
        self.corpus_md5_file = "corpus_md5.txt"

    def _calculate_corpus_md5(self, document_list: list) -> str:
        corpus_str = "\n".join([doc.strip() for doc in document_list])
        md5_obj = hashlib.md5(corpus_str.encode("utf-8"))
        return md5_obj.hexdigest()

    def _load_saved_corpus_md5(self) -> str:
        md5_path = os.path.join(self.index_dir, self.corpus_md5_file)
        if not os.path.exists(md5_path):
            return ""
        with open(md5_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def _save_corpus_md5(self, md5_str: str):
        md5_path = os.path.join(self.index_dir, self.corpus_md5_file)
        with open(md5_path, "w", encoding="utf-8") as f:
            f.write(md5_str)

    def text_embedding(self, text: str) -> np.ndarray:
        embedding = self.embedding_model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding

    def build_index(self, document_list: list, index_dir: str = "./faiss_index", index_type: str = "flat"):
        self.index_dir = index_dir
        current_corpus_md5 = self._calculate_corpus_md5(document_list)
        saved_corpus_md5 = self._load_saved_corpus_md5()

        index_path = os.path.join(index_dir, self.index_file)
        corpus_path = os.path.join(index_dir, self.corpus_file)
        need_rebuild = False

        if not os.path.exists(index_dir) or not os.path.exists(index_path) or not os.path.exists(corpus_path):
            need_rebuild = True
            print("FAISS索引文件缺失，需要重新构建索引...")
        elif current_corpus_md5 != saved_corpus_md5:
            need_rebuild = True
            print("语料已变更，需要重新构建FAISS索引...")
        else:
            need_rebuild = False
            print("FAISS索引存在且语料未变更，直接加载已有索引...")

        if not need_rebuild:
            self.load_index(index_dir)
            return

        self.corpus = [doc.strip() for doc in document_list]
        print(f"待处理文档数量：{len(self.corpus)}")

        print("开始文本向量化...")
        corpus_embeddings = self.embedding_model.encode(
            self.corpus,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        self.vector_dim = corpus_embeddings.shape[1]
        print(f"向量维度：{self.vector_dim}，向量矩阵形状：{corpus_embeddings.shape}")

        print("开始构建FAISS向量索引...")
        if index_type == "flat":
            self.index = faiss.IndexFlatIP(self.vector_dim)  # 内积索引，归一化后等价于余弦相似度
        elif index_type == "ivf_flat":
            nlist = 100
            self.index = faiss.IndexIVFFlat(
                faiss.IndexFlatIP(self.vector_dim),
                self.vector_dim,
                nlist,
                faiss.METRIC_INNER_PRODUCT
            )
            self.index.train(corpus_embeddings.astype(np.float32))
        else:
            raise ValueError(f"不支持的索引类型：{index_type}，可选flat/ivf_flat")

        self.index.add(corpus_embeddings.astype(np.float32))
        print(f"FAISS向量索引构建完成！索引包含 {self.index.ntotal} 个向量")

        self.save_index()
        self._save_corpus_md5(current_corpus_md5)

    def save_index(self):
        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)

        index_path = os.path.join(self.index_dir, self.index_file)
        faiss.write_index(self.index, index_path)

        corpus_path = os.path.join(self.index_dir, self.corpus_file)
        with open(corpus_path, "w", encoding="utf-8") as f:
            for doc in self.corpus:
                f.write(doc + "\n")

        print(f"FAISS索引保存完成：\n  - 向量索引：{index_path}\n  - 原始文档：{corpus_path}")

    def load_index(self, index_dir: str = "./faiss_index"):
        self.index_dir = index_dir
        index_path = os.path.join(index_dir, self.index_file)
        self.index = faiss.read_index(index_path)

        corpus_path = os.path.join(index_dir, self.corpus_file)
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.corpus = [line.strip() for line in f if line.strip()]

        self.vector_dim = self.index.d
        print(
            f"FAISS索引加载完成：\n  - 向量索引数量：{self.index.ntotal}\n  - 向量维度：{self.vector_dim}\n  - 原始文档数量：{len(self.corpus)}")

    def retrieve(self, query: str, top_k: int = 5) -> list:
        if self.index is None:
            raise ValueError("请先调用build_index构建FAISS索引（会自动加载已有索引）！")

        query_embedding = self.text_embedding(query).astype(np.float32)
        query_embedding = np.expand_dims(query_embedding, axis=0)

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i in range(top_k):
            vec_index = indices[0][i]
            if vec_index == -1:
                continue

            similarity_score = round(distances[0][i], 4)  # IndexFlatIP直接返回余弦相似度（0-1）
            results.append({
                "document": self.corpus[vec_index],
                "vector_index": vec_index,
                "similarity_score": similarity_score,
                "rank": i + 1,  # 排名从1开始
                "doc_id": f"DOC{vec_index:03d}"  # 与Whoosh的doc_id对齐
            })

        return results


# -------------------------- Whoosh关键词检索器（优化版，解决Token问题） --------------------------
class StandardJiebaTokenizer(Tokenizer):
    """符合Whoosh接口规范的jieba分词器，仅返回Token对象"""

    def __call__(self, value, positions=False, chars=False, keeporiginal=False,
                 removestops=True, start_pos=0, start_char=0, mode='', **kwargs):
        if not isinstance(value, str):
            value = str(value)
        words = jieba.lcut(value.strip())
        current_pos = start_pos
        current_char = start_char
        for word in words:
            if not word.strip():
                continue
            token = Token(
                text=word,
                pos=current_pos,
                startchar=current_char,
                endchar=current_char + len(word),
                stopped=False
            )
            current_pos += 1
            current_char += len(word)
            yield token


class JiebaAnalyzer(Analyzer):
    """标准Whoosh分析器，返回Token对象流"""

    def __init__(self):
        self.tokenizer = StandardJiebaTokenizer()

    def __call__(self, value, **kwargs):
        return self.tokenizer(value, **kwargs)


class WhooshKeywordRetriever:
    """Whoosh关键词检索器（支持索引复用、中文分词、Top-K检索）"""

    def __init__(self):
        self.index_dir = "./whoosh_index"
        self.schema = None
        self.ix = None

    def _init_schema(self):
        self.schema = Schema(
            doc_id=ID(unique=True, stored=True),
            title=TEXT(analyzer=JiebaAnalyzer(), stored=True),
            content=TEXT(analyzer=JiebaAnalyzer(), stored=True),
            doc_type=KEYWORD(stored=True)
        )

    def build_index(self, document_list):
        """
        :param document_list: 字典列表，格式：
                              {"doc_id": "DOC001", "title": "...", "content": "...", "doc_type": "..."}
        """
        self._init_schema()

        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)
            self.ix = create_in(self.index_dir, self.schema)
            print("Whoosh索引目录不存在，创建新索引并添加文档...")
            self._add_documents(document_list)
        else:
            self.ix = open_dir(self.index_dir)
            print("Whoosh索引已存在，直接打开已有索引...")

    def _add_documents(self, document_list):
        with self.ix.writer() as writer:
            for doc in document_list:
                writer.add_document(
                    doc_id=doc["doc_id"],
                    title=doc["title"],
                    content=doc["content"],
                    doc_type=doc["doc_type"]
                )
        print(f"成功添加 {len(document_list)} 篇文档到Whoosh索引！")

    def retrieve(self, query_str, top_k=5) -> list:
        if self.ix is None:
            raise ValueError("请先调用build_index构建Whoosh索引！")

        with self.ix.searcher(weighting=scoring.BM25F()) as searcher:
            parser = QueryParser("content", schema=self.ix.schema)
            query_str = str(query_str).strip()
            if not query_str:
                return []
            query = parser.parse(query_str)
            results = searcher.search(query, limit=top_k)

            retrieved_results = []
            for i, res in enumerate(results, 1):
                retrieved_results.append({
                    "document": res["content"],
                    "doc_id": res["doc_id"],
                    "title": res["title"],
                    "doc_type": res["doc_type"],
                    "similarity_score": round(res.score, 4),
                    "rank": i  # 排名从1开始
                })

        return retrieved_results


# -------------------------- RRF混合检索核心实现 --------------------------
def reciprocal_rank_fusion(vector_results: list, keyword_results: list, k: int = 6, top_k: int = 5) -> list:
    """
    倒数排名融合（RRF）：无需归一化，仅通过排名实现混合检索
    :param vector_results: FAISS向量检索结果列表（含doc_id和rank）
    :param keyword_results: Whoosh关键词检索结果列表（含doc_id和rank）
    :param k: 平滑系数（经验值6-10，避免排名第1的文档权重过高）
    :param top_k: 融合后返回的Top-K文档数量
    :return: 融合排序后的结果列表
    """
    # 构建文档ID到排名/内容的映射
    doc_map = {}

    # 录入FAISS向量检索结果
    for res in vector_results:
        doc_id = res["doc_id"]
        doc_map[doc_id] = {
            "doc_id": doc_id,
            "document": res["document"],
            "vector_rank": res["rank"],
            "whoosh_rank": None,
            "vector_score": res.get("similarity_score", 0.0),
            "whoosh_score": None,
            "rrf_score": 0.0
        }

    # 录入Whoosh关键词检索结果
    for res in keyword_results:
        doc_id = res["doc_id"]
        if doc_id in doc_map:
            doc_map[doc_id]["whoosh_rank"] = res["rank"]
            doc_map[doc_id]["whoosh_score"] = res.get("similarity_score", 0.0)
            doc_map[doc_id]["title"] = res.get("title", "")
            doc_map[doc_id]["doc_type"] = res.get("doc_type", "")
        else:
            doc_map[doc_id] = {
                "doc_id": doc_id,
                "document": res["document"],
                "vector_rank": None,
                "whoosh_rank": res["rank"],
                "vector_score": None,
                "whoosh_score": res.get("similarity_score", 0.0),
                "title": res.get("title", ""),
                "doc_type": res.get("doc_type", ""),
                "rrf_score": 0.0
            }

    # 计算RRF得分
    for doc_id in doc_map:
        rrf_sum = 0.0
        # 向量检索排名贡献（存在则计算）
        if doc_map[doc_id]["vector_rank"] is not None:
            rrf_sum += 1 / (k + doc_map[doc_id]["vector_rank"])
        # 关键词检索排名贡献（存在则计算）
        if doc_map[doc_id]["whoosh_rank"] is not None:
            rrf_sum += 1 / (k + doc_map[doc_id]["whoosh_rank"])
        doc_map[doc_id]["rrf_score"] = round(rrf_sum, 4)

    # 按RRF得分降序排序，返回Top-K
    sorted_hybrid_results = sorted(
        doc_map.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )[:top_k]

    # 补充融合后的排名
    for i, res in enumerate(sorted_hybrid_results, 1):
        res["hybrid_rank"] = i

    return sorted_hybrid_results


# -------------------------- 混合检索测试示例 --------------------------
if __name__ == "__main__":
    # 1. 准备测试文档（统一格式，同时适配FAISS和Whoosh）
    test_documents = [
        {
            'doc_id': 'DOC001',
            'title': 'RAG核心组件介绍',
            'content': 'RAG（检索增强生成）的核心组件包括知识库、向量数据库、大语言模型，BM25是常用的关键词检索算法。',
            'doc_type': '技术文档'
        },
        {
            'doc_id': 'DOC002',
            'title': 'BM25算法优势',
            'content': 'BM25算法是TF-IDF的改进版，引入了文档长度归一化和词频饱和机制，适合全文检索场景。',
            'doc_type': '技术文档'
        },
        {
            'doc_id': 'DOC003',
            'title': 'Windows下Python实现BM25',
            'content': 'Windows系统下使用Python实现BM25检索，需要安装jieba和rank_bm25库，或直接使用Whoosh简化开发。',
            'doc_type': '实操手册'
        }
    ]
    # test_documents = [
    #     {
    #         "doc_id": "DOC001",
    #         "title": "RAG核心组件介绍",
    #         "content": "RAG（检索增强生成）的核心组件包括知识库、向量数据库、大语言模型，BM25是常用的关键词检索算法，FAISS是主流向量检索库。",
    #         "doc_type": "技术文档"
    #     },
    #     {
    #         "doc_id": "DOC002",
    #         "title": "BM25算法优势",
    #         "content": "BM25算法是TF-IDF的改进版，引入了文档长度归一化和词频饱和机制，适合全文检索场景，常与向量检索结合实现Hybrid RAG。",
    #         "doc_type": "技术文档"
    #     },
    #     {
    #         "doc_id": "DOC003",
    #         "title": "Windows下Python实现BM25与FAISS",
    #         "content": "Windows系统下使用Python实现BM25检索需要安装jieba和rank_bm25库，实现FAISS向量检索需要安装faiss-cpu和sentence-transformers库。",
    #         "doc_type": "实操手册"
    #     },
    #     {
    #         "doc_id": "DOC004",
    #         "title": "大语言模型幻觉解决方案",
    #         "content": "大语言模型容易产生幻觉，通过RAG检索外部知识库（结合BM25关键词检索和FAISS向量检索）可以有效减少虚假信息生成。",
    #         "doc_type": "技术文档"
    #     },
    #     {
    #         "doc_id": "DOC005",
    #         "title": "Hybrid RAG最佳实践",
    #         "content": "混合检索（Hybrid RAG）通过RRF融合BM25和FAISS的检索结果，兼顾关键词精准匹配和语义相似性，大幅提升检索效果。",
    #         "doc_type": "技术文档"
    #     }
    # ]

    # 2. 提取纯文本列表（用于FAISS向量索引，FAISS仅需文本内容）
    faiss_corpus = [doc["content"] for doc in test_documents]

    # 3. 初始化并构建检索器索引
    # FAISS向量检索器
    faiss_retriever = FAISSVectorRetriever(model_name=r'C:\Users\gaohu\aiModel\text2vec-base-chinese')
    faiss_retriever.build_index(faiss_corpus, index_dir="./faiss_rag_index", index_type="flat")

    # Whoosh关键词检索器
    whoosh_retriever = WhooshKeywordRetriever()
    whoosh_retriever.build_index(test_documents)

    # 4. 定义用户查询
    # user_queries = [
    #     "RAG 核心组件与混合检索",
    #     "BM25 算法优势与实现",
    #     "Windows Python 实现 FAISS",
    #     "如何减少大语言模型幻觉",
    #     "Hybrid RAG 最佳实践"
    # ]
    user_queries = ["RAG 核心组件", "BM25 算法", "Windows Python"]

    # 5. 执行混合检索（FAISS + Whoosh + RRF）
    for query_idx, query in enumerate(user_queries, 1):
        print(f"\n{'=' * 80}")
        print(f"========== 第 {query_idx} 个查询：{query} ==========")
        print(f"{'=' * 80}")

        # 5.1 分别执行向量检索和关键词检索
        top_k = 5  # 单个检索返回Top-5结果
        vector_results = faiss_retriever.retrieve(query, top_k=top_k)
        keyword_results = whoosh_retriever.retrieve(query, top_k=top_k)

        # 5.2 输出单独检索结果
        print("\n--- FAISS向量检索Top-3结果 ---")
        for res in vector_results[:3]:
            print(f"排名 {res['rank']} | 相似度 {res['similarity_score']} | 文档ID {res['doc_id']}")
            print(f"文档内容：{res['document'][:80]}...")

        print("\n--- Whoosh关键词检索Top-3结果 ---")
        for res in keyword_results[:3]:
            print(
                f"排名 {res['rank']} | BM25得分 {res['similarity_score']} | 文档ID {res['doc_id']} | 标题 {res['title']}")
            print(f"文档内容：{res['document'][:80]}...")

        # 5.3 RRF混合融合
        hybrid_results = reciprocal_rank_fusion(
            vector_results=vector_results,
            keyword_results=keyword_results,
            k=6,  # 平滑系数
            top_k=5  # 融合后返回Top-5
        )

        # 5.4 输出混合检索结果
        print("\n--- RRF混合检索Top-5结果（最终推荐） ---")
        for res in hybrid_results:
            print(f"混合排名 {res['hybrid_rank']} | RRF得分 {res['rrf_score']}")
            print(
                f"文档ID {res['doc_id']} | 标题 {res.get('title', '无标题')} | 文档类型 {res.get('doc_type', '无类型')}")
            print(f"向量排名 {res['vector_rank'] or '无'} | 关键词排名 {res['whoosh_rank'] or '无'}")
            print(f"文档内容：{res['document'][:100]}...")
            print("-" * 60)
