from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID, KEYWORD
from whoosh.analysis import Tokenizer, Token, Analyzer
from whoosh.qparser import QueryParser
from whoosh import scoring
import os
import jieba

# 1. 严格实现Whoosh标准Tokenizer接口（确保返回Token对象，而非字符串）
class StandardJiebaTokenizer(Tokenizer):
    """符合Whoosh接口规范的jieba分词器，仅返回Token对象，杜绝字符串"""
    def __call__(self, value, positions=False, chars=False, keeporiginal=False,
                 removestops=True, start_pos=0, start_char=0, mode='', **kwargs):
        # 第一步：确保输入是纯字符串，避免异常
        if not isinstance(value, str):
            value = str(value)
        # 第二步：jieba分词，获取词汇列表
        words = jieba.lcut(value.strip())
        # 第三步：遍历词汇，生成标准Token对象（必须包含stopped属性）
        current_pos = start_pos
        current_char = start_char
        for word in words:
            if not word.strip():  # 过滤空字符串
                continue
            # 构建Whoosh标准Token对象（核心：显式初始化所有必要属性）
            token = Token(
                text=word,  # 分词后的词汇文本
                pos=current_pos,  # 位置信息
                startchar=current_char,  # 字符起始位置
                endchar=current_char + len(word),  # 字符结束位置
                stopped=False  # 显式设置stopped属性，默认非停用词
            )
            # 更新位置信息
            current_pos += 1
            current_char += len(word)
            # 返回Token对象（而非字符串）
            yield token

# 2. 封装标准Analyzer（兼容Whoosh分析器接口）
class JiebaAnalyzer(Analyzer):
    """标准Whoosh分析器，返回Token对象流，无字符串泄露"""
    def __init__(self):
        self.tokenizer = StandardJiebaTokenizer()

    def __call__(self, value, **kwargs):
        # 直接返回Tokenizer生成的Token流，不额外转换为字符串
        return self.tokenizer(value, **kwargs)

# 3. 检索器主类（无其他隐含问题）
class WhooshKeywordRetriever:
    """Whoosh关键词检索器（彻底解决Token/字符串类型错误）"""
    def __init__(self):
        self.index_dir = "./whoosh_index_new"
        self.schema = None
        self.ix = None

    def _init_schema(self):
        """初始化Schema，使用标准JiebaAnalyzer，确保字段与Token流兼容"""
        self.schema = Schema(
            doc_id=ID(unique=True, stored=True),  # ID字段：无需分词，直接存储
            title=TEXT(analyzer=JiebaAnalyzer(), stored=True),  # 用标准分析器
            content=TEXT(analyzer=JiebaAnalyzer(), stored=True),  # 用标准分析器
            doc_type=KEYWORD(stored=True)  # KEYWORD字段：无需分词
        )

    def build_index(self, document_list):
        """构建索引（兼容你的字典列表文档）"""
        self._init_schema()

        # 索引目录处理
        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)
            self.ix = create_in(self.index_dir, self.schema)
            print("索引目录不存在，创建新索引并添加文档...")
            self._add_documents(document_list)
        else:
            self.ix = open_dir(self.index_dir)
            print("索引已存在，直接打开已有索引...")

    def _add_documents(self, document_list):
        """批量添加文档（严格管理writer，确保Token流正常）"""
        # 必须用with语句管理writer，确保分析器资源正确初始化
        with self.ix.writer() as writer:
            for doc in document_list:
                # 仅做简单校验，无需强制转换（你的文档格式已正确）
                writer.add_document(
                    doc_id=doc["doc_id"],
                    title=doc["title"],
                    content=doc["content"],
                    doc_type=doc["doc_type"]
                )
        print(f"成功添加 {len(document_list)} 篇文档，无字符串/Token类型错误！")

    def retrieve(self, query_str, top_k=5):
        """执行检索"""
        if self.ix is None:
            raise ValueError("请先调用build_index构建/打开索引！")

        with self.ix.searcher(weighting=scoring.BM25F()) as searcher:
            parser = QueryParser("content", schema=self.ix.schema)
            query = parser.parse(query_str.strip())
            results = searcher.search(query, limit=top_k)

            # 整理结果
            retrieved_results = []
            for i, res in enumerate(results, 1):
                retrieved_results.append({
                    "rank": i,
                    "similarity_score": round(res.score, 4),
                    "doc_id": res["doc_id"],
                    "title": res["title"],
                    "content": res["content"],
                    "doc_type": res["doc_type"],
                    "highlighted_content": res.highlights("content")
                })
        return retrieved_results

# -------------------------- 测试你的原始文档（无需任何修改）--------------------------
if __name__ == "__main__":
    # 你的原始字典列表文档（完全不变）
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

    # 初始化并运行
    whoosh_retriever = WhooshKeywordRetriever()
    whoosh_retriever.build_index(test_documents)

    # 检索测试
    for i, query in enumerate(["RAG 核心组件", "BM25 算法", "Windows Python"], 1):
        print(f"\n========== 第 {i} 次查询：{query} ==========")
        results = whoosh_retriever.retrieve(query, top_k=2)
        for res in results:
            print(f"排名 {res['rank']} | 得分 {res['similarity_score']}")
            print(f"文档ID：{res['doc_id']} | 标题：{res['title']}")
            print(f"高亮内容：{res['highlighted_content'] or res['content'][:60]}...")
            print("-" * 70)