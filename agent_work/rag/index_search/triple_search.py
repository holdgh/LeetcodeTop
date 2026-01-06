import os.path

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.embeddings.text2vec import Text2vecEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever  # 新增：融合BM25和向量检索
from sentence_transformers import SentenceTransformer


if __name__ == '__main__':
    # ====================== 1. 初始化核心模型（补全embed_model使用） ======================
    # 嵌入模型（用于语义检索精排）Qwen/Qwen3-Embedding-0.6B-GGUF
    embed_model = Text2vecEmbeddings(model_name_or_path=r'C:\Users\gaohu\aiModel\text2vec-base-chinese')
    # 重排模型（用于最终精准排序）Qwen/Qwen3-Reranker-0.6B
    # rerank_model = HuggingFaceCrossEncoder(model_name_or_path="BAAI/bge-reranker-large")

    # ====================== 2. 准备知识库文档（示例数据） ======================
    # 模拟语义chunk化后的知识库文档
    # enhanced_docs = [
    #     Document(
    #         page_content="语义感知的动态Chunking是RAG检索优化的核心方案，优先按中文语义分隔符切分文档，保证每个Chunk是完整语义单元。",
    #         metadata={"chunk_id": 1, "type": "text"}
    #     ),
    #     Document(
    #         page_content="BM25是关键词检索算法，基于词频和逆文档频率计算相似度，适合快速粗排过滤无关语料。",
    #         metadata={"chunk_id": 2, "type": "text"}
    #     ),
    #     Document(
    #         page_content="交叉编码器重排模型可计算「问题-语料」的精准匹配度，提升相似语料的区分能力。",
    #         metadata={"chunk_id": 3, "type": "text"}
    #     )
    # ]
    from agent_work.rag.chunk.lang_chain_chunk_demo import chunk_the_txt
    enhanced_docs = chunk_the_txt(r'../chunk/first_hui.txt')

    # ====================== 3. 构建三级检索器（补全embed_model核心逻辑） ======================
    # 步骤1：粗排 - BM25关键词检索（快速召回候选集）
    bm25_retriever = BM25Retriever.from_documents(enhanced_docs)
    # bm25_retriever.k = 50  # 粗排召回50条，过滤完全无关内容
    bm25_retriever.k = 5  # 粗排召回5条，过滤完全无关内容

    # 步骤2：精排 - 向量语义检索（基于embed_model，计算语义相似度）
    vector_index_name = 'xiYouJi'
    vector_path_name = r"C:\Users\gaohu\aiPyProject\LeetcodeTop\agent_work\rag\index\vector"
    if not os.path.exists(os.path.join(vector_path_name, vector_index_name + ".faiss")):
        # ① 构建FAISS向量库（核心：用embed_model将文档转为向量存储）
        vector_db = FAISS.from_documents(enhanced_docs, embed_model)
        vector_db.save_local(folder_path=vector_path_name, index_name=vector_index_name)
    else:
        vector_db = FAISS.load_local(folder_path=vector_path_name, embeddings=embed_model, index_name=vector_index_name, allow_dangerous_deserialization=True)
    # ② 构建向量检索器（精排，从粗排结果中进一步筛选）
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 5})  # 精排保留5条

    # 步骤3：融合粗排+精排结果（可选：若需同时用BM25和向量检索的结果）
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.3, 0.7]  # 权重可调：向量检索占比更高，优先语义匹配
    )  # 混合检索，内部采用了rrf评分机制 叠加【权重*1/(平衡因子+顺序)】

    # 步骤4：重排 - 交叉编码器精准排序（最终筛选高匹配度语料）
    final_retriever = ensemble_retriever
    # compressor = CrossEncoderReranker(model=rerank_model, top_n=3)
    # final_retriever = ContextualCompressionRetriever(
    #     base_compressor=compressor,
    #     base_retriever=ensemble_retriever  # 基于融合后的结果重排
    # )

    # ====================== 4. 执行检索（验证embed_model生效） ======================
    # query = "如何解决RAG中语义切分导致的检索精度问题？"
    query = "孙悟空是哪里人？"
    # 最终检索结果（三级过滤后，仅返回Top3高匹配度语料）
    final_docs = final_retriever.invoke(query)

    # 打印检索结果
    # print("===== 三级检索最终结果 =====")
    print(f"{8*'='} 两级检索最终结果 {8*'='}")
    for i, doc in enumerate(final_docs, 1):
        print(f"第{i}条语料：{doc.page_content}")
        print(f"语料元信息：{doc.metadata}")
        print("-" * 50)


# if __name__ == '__main__':
#     from langchain_community.embeddings.text2vec import Text2vecEmbeddings
#
#     embedding = Text2vecEmbeddings(model_name_or_path=r'C:\Users\gaohu\aiModel\text2vec-base-chinese')
#     print(embedding.embed_documents([
#         "This is a CoSENT(Cosine Sentence) model.",
#         "It maps sentences to a 768 dimensional dense vector space.",
#     ]))
#     print(embedding.embed_query(
#         "It can be used for text matching or semantic search."
#     ))