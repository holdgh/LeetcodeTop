# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters.markdown import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from langchain_core.documents import Document


# txt文档切分
def chunk_the_txt(txt_file_path: str) -> list[Document]:
    # 1. 加载文档
    loader = TextLoader(txt_file_path, encoding="utf-8")
    documents = loader.load()

    # 2. 语义感知切分：优先按句子分割，避免切断完整语义
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # 最大chunk长度
        chunk_overlap=50,  # 相邻chunk重叠部分（保持上下文关联）
        separators=["\n\n", "\n", "。", "！", "？", "；", "，"],  # 优先按中文语义分隔符切分
        length_function=len  # 字符长度计算
    )

    # 3. 生成语义完整的chunk
    split_docs = text_splitter.split_documents(documents)
    # 每个chunk都是完整语义单元，避免单段落被切散
    # for doc in split_docs:
    #     print(f"Chunk内容：{doc.page_content[:100]}...")
    #     print(f"Chunk元信息：{doc.metadata}")
    # 分块之间的上下文语义扩展

    # 1. 为切分后的文本分块添加前后chunk元信息
    enhanced_docs = []
    for i, doc in enumerate(split_docs):
        doc.metadata["chunk_id"] = i
        doc.metadata["prev_chunk_id"] = i - 1 if i > 0 else None
        doc.metadata["next_chunk_id"] = i + 1 if i < len(split_docs) - 1 else None
        enhanced_docs.append(doc)

    return enhanced_docs


# 2. 检索后扩展上下文【对于检索结果追加前后chunk】
def extend_context(retrieved_docs, all_docs):
    extended_docs = []
    for doc in retrieved_docs:
        current_id = doc.metadata["chunk_id"]
        # 拉取前后各1个chunk【从所在文档的全量chunk中取出当前检索chunk的前后chunk，进行换行拼接】
        prev_doc = next((d for d in all_docs if d.metadata["chunk_id"] == doc.metadata["prev_chunk_id"]), None)
        next_doc = next((d for d in all_docs if d.metadata["chunk_id"] == doc.metadata["next_chunk_id"]), None)
        # 拼接上下文
        full_content = ""
        if prev_doc:
            full_content += prev_doc.page_content + "\n"
        full_content += doc.page_content
        if next_doc:
            full_content += "\n" + next_doc.page_content
        # 生成新文档
        extended_docs.append(Document(page_content=full_content, metadata=doc.metadata))
    return extended_docs


if __name__ == '__main__':
    enhanced_docs = chunk_the_txt("first_hui.txt")
    # 模拟检索结果
    retrieved_docs = [d for d in enhanced_docs if d.metadata["chunk_id"] == 2]
    # 扩展上下文
    final_context_docs = extend_context(retrieved_docs, enhanced_docs)
