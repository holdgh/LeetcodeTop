from whoosh.index import open_dir
from whoosh.reading import IndexReader
from RRF_hybrid_search_with_database import JiebaAnalyzer, StandardJiebaTokenizer


# class StandardJiebaTokenizer(Tokenizer):
#     """符合Whoosh接口规范的jieba分词器，仅返回Token对象"""
#
#     def __call__(self, value, positions=False, chars=False, keeporiginal=False,
#                  removestops=True, start_pos=0, start_char=0, mode='', **kwargs):
#         if not isinstance(value, str):
#             value = str(value)
#         words = jieba.lcut(value.strip())
#         current_pos = start_pos
#         current_char = start_char
#         for word in words:
#             if not word.strip():
#                 continue
#             token = Token(
#                 text=word,
#                 pos=current_pos,
#                 startchar=current_char,
#                 endchar=current_char + len(word),
#                 stopped=False
#             )
#             current_pos += 1
#             current_char += len(word)
#             yield token

if __name__ == '__main__':
    # 1. 打开已创建的索引
    ix = open_dir(r'C:\Users\gaohu\aiPyProject\LeetcodeTop\agent_work\rag\whoosh_index')
    reader: IndexReader = ix.reader()  # 获取索引读取器，核心工具

    # ====================== 方式1：查看所有字段的元信息（关键词库所属字段） ======================
    print("===== 索引字段元信息（关键词库所属字段） =====")
    for field_name in reader.field_names():
        field = reader.schema[field_name]
        print(f"字段名：{field_name}")
        print(f"字段类型：{type(field).__name__}")
        print(f"是否分词：{field.indexed}")
        print(f"是否存储原文：{field.stored}")
        print("-" * 50)

    # ====================== 方式2：遍历指定字段的所有关键词（分词后的词条） ======================
    print("\n===== 查看content字段的所有关键词（词条） =====")
    target_field = "content"  # 要查看的字段（即你要获取关键词库的字段）

    # 使用 reader.terms() 遍历指定字段的所有词条（按字母排序）
    # 参数说明：fieldname=字段名，prefix=可选，用于过滤以指定前缀开头的词条
    for term, freq in reader.terms(fieldname=target_field):
        # term：分词后的关键词（词条）；freq：该词条在索引中出现的总次数
        print(f"关键词：{term}，出现次数：{freq}")

    # ====================== 方式3：查看单个关键词的详细信息（文档映射、位置等） ======================
    print("\n===== 查看单个关键词的详细信息 =====")
    target_term = "whoosh"  # 要查询的关键词
    # 使用 reader.postings() 获取该词条的倒排索引信息
    postings = reader.postings(target_field, target_term)

    # 遍历该关键词对应的所有文档
    for doc_num in postings.all_ids():
        # 根据文档编号获取文档内容
        doc = reader.stored_fields(doc_num)
        print(f"所属文档ID：{doc['doc_id']}")
        print(f"文档内容：{doc['content']}")
        # 查看该关键词在文档中的位置（起始偏移量、长度）
        positions = list(postings.positions(doc_num))
        print(f"关键词在文档中的位置：{positions}")
        print("-" * 50)

    # ====================== 方式4：查看索引中的所有文档（辅助验证关键词所属文档） ======================
    print("\n===== 索引中的所有文档 =====")
    for doc_num in range(reader.doc_count()):
        doc = reader.stored_fields(doc_num)
        print(f"文档编号：{doc_num}，文档ID：{doc['doc_id']}，内容：{doc['content']}")

    # ====================== 方式5：统计关键词库规模（总词条数、唯一词条数） ======================
    print("\n===== 关键词库规模统计 =====")
    # 总词条数（包含重复）
    total_terms = sum(reader.term_frequency(field, term) for field in reader.field_names() if reader.schema[field].indexed for term, _ in reader.terms(field))
    # 唯一词条数（去重）
    unique_terms = sum(1 for field in reader.field_names() if reader.schema[field].indexed for _, _ in reader.terms(field))
    print(f"索引中总词条数（含重复）：{total_terms}")
    print(f"索引中唯一词条数（关键词库大小）：{unique_terms}")

    # 关闭读取器（可选，Python会自动回收）
    reader.close()