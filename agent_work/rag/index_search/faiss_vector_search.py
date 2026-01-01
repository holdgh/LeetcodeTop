import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
import hashlib


class FAISSVectorRetriever:
    """FAISS向量检索器（新增索引存在性校验，避免重复构建）"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        初始化向量检索器
        :param model_name: 文本向量化模型（Sentence-BERT系列）
        """
        # 文本向量化模型
        self.embedding_model = SentenceTransformer(model_name)
        # FAISS索引对象
        self.index = None
        # 原始文档列表（映射向量索引与原始文档）
        self.corpus = []
        # 向量维度（由向量化模型决定）
        self.vector_dim = None
        # 索引相关路径
        self.index_dir = None
        self.index_file = "vector_index.faiss"
        self.corpus_file = "corpus.txt"
        self.corpus_md5_file = "corpus_md5.txt"  # 用于存储语料MD5值，校验语料是否变更

    def _calculate_corpus_md5(self, document_list: list) -> str:
        """
        计算语料的MD5哈希值，用于校验语料是否发生变更
        :param document_list: 原始文档列表
        :return: 语料的MD5字符串
        """
        # 将所有文档拼接为一个字符串，计算MD5
        corpus_str = "\n".join([doc.strip() for doc in document_list])
        md5_obj = hashlib.md5(corpus_str.encode("utf-8"))
        return md5_obj.hexdigest()

    def _load_saved_corpus_md5(self) -> str:
        """加载已保存的语料MD5值"""
        md5_path = os.path.join(self.index_dir, self.corpus_md5_file)
        if not os.path.exists(md5_path):
            return ""
        with open(md5_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def _save_corpus_md5(self, md5_str: str):
        """保存当前语料的MD5值"""
        md5_path = os.path.join(self.index_dir, self.corpus_md5_file)
        with open(md5_path, "w", encoding="utf-8") as f:
            f.write(md5_str)

    def text_embedding(self, text: str) -> np.ndarray:
        """单文本向量化：将中文文本转换为稠密向量"""
        embedding = self.embedding_model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True  # 归一化向量，提升余弦相似度计算精度
        )
        return embedding

    def build_index(self, document_list: list, index_dir: str = "./faiss_index", index_type: str = "flat"):
        """
        构建FAISS向量索引（新增校验：索引不存在/语料变更时才构建）
        :param document_list: 原始文档列表（每个元素为一篇文档/文本块）
        :param index_dir: 索引保存目录
        :param index_type: 索引类型（flat=精确检索，ivf_flat=近似检索）
        """
        self.index_dir = index_dir
        current_corpus_md5 = self._calculate_corpus_md5(document_list)
        saved_corpus_md5 = self._load_saved_corpus_md5()

        # 校验条件：索引目录不存在 / 索引文件缺失 / 语料MD5不匹配（语料变更）
        index_path = os.path.join(index_dir, self.index_file)
        corpus_path = os.path.join(index_dir, self.corpus_file)
        need_rebuild = False

        if not os.path.exists(index_dir) or not os.path.exists(index_path) or not os.path.exists(corpus_path):
            need_rebuild = True
            print("索引文件缺失，需要重新构建索引...")
        elif current_corpus_md5 != saved_corpus_md5:
            need_rebuild = True
            print("语料已变更，需要重新构建索引...")
        else:
            need_rebuild = False
            print("索引存在且语料未变更，直接加载已有索引...")

        # 无需重建：直接加载索引
        if not need_rebuild:
            self.load_index(index_dir)
            return

        # 需要重建：执行索引构建流程
        self.corpus = [doc.strip() for doc in document_list]
        print(f"待处理文档数量：{len(self.corpus)}")

        # 批量向量化（效率高于单条转换）
        print("开始文本向量化...")
        corpus_embeddings = self.embedding_model.encode(
            self.corpus,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True  # 显示进度条（大规模数据时友好）
        )
        self.vector_dim = corpus_embeddings.shape[1]
        print(f"向量维度：{self.vector_dim}，向量矩阵形状：{corpus_embeddings.shape}")

        # 构建FAISS索引
        print("开始构建FAISS向量索引...")
        if index_type == "flat":
            # Flat索引（暴力检索，精确匹配，适合小规模数据<10万条）
            # 若需余弦相似度，替换为IndexFlatIP（内积，向量归一化后等价于余弦相似度）
            self.index = faiss.IndexFlatL2(self.vector_dim)
        elif index_type == "ivf_flat":
            # IVF_FLAT索引（近似检索，适合大规模数据>10万条）
            nlist = 100  # 聚类中心数量（经验值：数据量^(1/3)）
            self.index = faiss.IndexIVFFlat(
                faiss.IndexFlatL2(self.vector_dim),
                self.vector_dim,
                nlist,
                faiss.METRIC_L2
            )
            # 训练索引（近似检索必须先训练）
            self.index.train(corpus_embeddings.astype(np.float32))
        else:
            raise ValueError(f"不支持的索引类型：{index_type}，可选flat/ivf_flat")

        # 添加向量到索引（FAISS要求float32类型）
        self.index.add(corpus_embeddings.astype(np.float32))
        print(f"FAISS向量索引构建完成！索引包含 {self.index.ntotal} 个向量")

        # 保存索引、语料和MD5值
        self.save_index()
        self._save_corpus_md5(current_corpus_md5)

    def save_index(self):
        """保存向量索引与文档映射（内部调用，无需手动执行）"""
        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)

        # 保存FAISS索引
        index_path = os.path.join(self.index_dir, self.index_file)
        faiss.write_index(self.index, index_path)

        # 保存原始文档（txt格式，映射向量索引）
        corpus_path = os.path.join(self.index_dir, self.corpus_file)
        with open(corpus_path, "w", encoding="utf-8") as f:
            for doc in self.corpus:
                f.write(doc + "\n")

        print(f"索引保存完成：\n  - 向量索引：{index_path}\n  - 原始文档：{corpus_path}")

    def load_index(self, index_dir: str = "./faiss_index"):
        """加载持久化的向量索引（原有功能，保持不变）"""
        self.index_dir = index_dir
        # 加载FAISS索引
        index_path = os.path.join(index_dir, self.index_file)
        self.index = faiss.read_index(index_path)

        # 加载原始文档
        corpus_path = os.path.join(index_dir, self.corpus_file)
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.corpus = [line.strip() for line in f if line.strip()]

        # 获取向量维度
        self.vector_dim = self.index.d
        print(
            f"索引加载完成：\n  - 向量索引数量：{self.index.ntotal}\n  - 向量维度：{self.vector_dim}\n  - 原始文档数量：{len(self.corpus)}")

    def retrieve(self, query: str, top_k: int = 5) -> list:
        """
        执行向量检索（返回Top-K相似文档）
        :param query: 用户查询文本（中文）
        :param top_k: 返回最相似的前k篇文档
        :return: 检索结果列表
        """
        if self.index is None:
            raise ValueError("请先调用build_index构建索引（会自动加载已有索引）！")

        # 步骤1：查询文本向量化
        query_embedding = self.text_embedding(query).astype(np.float32)
        query_embedding = np.expand_dims(query_embedding, axis=0)  # 转换为2D数组（适配FAISS输入格式）

        # 步骤2：执行向量检索（返回距离与索引）
        distances, indices = self.index.search(query_embedding, top_k)

        # 步骤3：整理检索结果（转换距离为相似度得分，映射原始文档）
        results = []
        for i in range(top_k):
            vec_index = indices[0][i]
            if vec_index == -1:  # 无匹配结果（索引中不存在有效向量）
                continue

            # 转换距离为相似度（0-1区间，便于后续混合检索）
            # L2距离转换为相似度：similarity = 1 / (1 + distance)
            # 若使用IndexFlatIP（内积），直接使用distance作为相似度即可
            distance = distances[0][i]
            similarity_score = round(1 / (1 + distance), 4)

            results.append({
                "document": self.corpus[vec_index],  # 原始文档内容
                "vector_index": vec_index,  # 向量索引
                "distance": round(distance, 4),  # 原始距离值
                "similarity_score": similarity_score,  # 归一化相似度得分（0-1）
                "rank": i + 1  # 排名（从1开始）
            })

        return results


# -------------------------- 测试使用示例 --------------------------
if __name__ == "__main__":
    # 1. 准备测试文档（模拟RAG知识库文本块）
    test_documents = [
        "RAG（检索增强生成）的核心组件包括知识库、向量数据库、大语言模型，FAISS是常用的向量检索库。",
        "FAISS由Facebook开源，支持高效的近似最近邻检索，适合大规模向量数据处理。",
        "大语言模型容易产生幻觉，通过RAG检索外部知识库可以有效减少虚假信息生成。",
        "Windows系统下使用Python实现FAISS向量检索，需要安装faiss-cpu和sentence-transformers库。",
        "向量检索注重语义相似性，BM25注重关键词匹配，两者结合可实现Hybrid RAG提升效果。",
        "中文文本向量检索需要先进行向量化处理，Sentence-BERT是常用的中文文本向量化模型。"
    ]

    # 2. 初始化FAISS向量检索器
    vector_retriever = FAISSVectorRetriever(model_name=r'C:\Users\gaohu\aiModel\text2vec-base-chinese')

    # 3. 构建索引（首次运行会构建并保存，后续运行语料不变则直接加载已有索引）
    vector_retriever.build_index(
        document_list=test_documents,
        index_dir="./faiss_rag_index",
        index_type="flat"
    )

    # 4. 执行向量检索（模拟用户查询）
    user_queries = [
        "FAISS 向量检索的优势",
        "RAG 核心组件有哪些",
        "Windows 下 Python 实现 FAISS",
        "如何减少大语言模型幻觉"
    ]

    # 5. 遍历查询并输出结果
    for i, query in enumerate(user_queries, 1):
        print(f"\n========== 第 {i} 个查询：{query} ==========")
        results = vector_retriever.retrieve(query, top_k=2)  # 返回前2篇最相似文档
        for j, res in enumerate(results, 1):
            print(f"排名 {j}（相似度：{res['similarity_score']} | 距离：{res['distance']}）：")
            print(f"文档内容：{res['document']}")
            print(f"向量索引：{res['vector_index']}")
            print("-" * 60)
