# -*- coding: utf-8 -*-
"""
中文情感分类噪声样本检测脚本
功能：自动识别标签标反的噪声样本（如负向文本标1/正向、正向文本标0/负向）
适用：商品评价、酒店评论等短文本情感分类数据集
输入：csv格式数据集（需包含text列、label列，label=0负向/1正向）
输出：噪声样本列表+可视化统计
"""
import pandas as pd
import jieba
import warnings

warnings.filterwarnings("ignore")

# ====================== 1. 配置参数（新手只需改这里） ======================
INPUT_CSV_PATH = "../../jd_comment/dev.csv"  # 你的数据集路径（替换成实际路径）
OUTPUT_NOISE_CSV = "../../jd_comment/noise_samples.csv"  # 噪声样本保存路径
TEXT_COL = "sentence"  # 文本列名（无需改，除非你的列名不是text）
LABEL_COL = "label"  # 标签列名（无需改，除非你的列名不是label）

# 情感关键词库（可根据你的场景补充，比如酒店评论加"隔音差""床品舒服"等）
POSITIVE_KEYWORDS = [
    "好", "棒", "优秀", "满意", "舒服", "划算", "推荐", "惊喜", "完美", "值",
    "方便", "快捷", "贴心", "靠谱", "赞", "给力", "耐用", "美观", "实用"
]
NEGATIVE_KEYWORDS = [
    "差", "烂", "糟糕", "不满意", "坑", "骗人", "差", "贵", "慢", "破",
    "垃圾", "恶心", "失望", "不行", "麻烦", "粗糙", "失灵", "漏", "坏"
]


# ====================== 2. 核心检测函数 ======================
def calculate_sentiment_score(text):
    """计算文本情感得分：正向词数 - 负向词数"""
    words = jieba.lcut(text)
    pos_count = sum([1 for word in words if word in POSITIVE_KEYWORDS])
    neg_count = sum([1 for word in words if word in NEGATIVE_KEYWORDS])
    return pos_count - neg_count


def detect_noise_sample(row):
    """检测单条样本是否为噪声：情感得分与标签矛盾"""
    text = str(row[TEXT_COL]).strip()
    label = row[LABEL_COL]
    score = calculate_sentiment_score(text)

    # 判定规则：
    # 1. 正向得分高但标签为0（负向）→ 噪声
    # 2. 负向得分高但标签为1（正向）→ 噪声
    # 3. 无情感词 → 中性样本（不判定为噪声）
    if score > 0 and label == 0:
        return True, "正向文本标负向"
    elif score < 0 and label == 1:
        return True, "负向文本标正向"
    else:
        return False, "正常样本/中性样本"


def batch_detect_noise(df):
    """批量检测噪声样本"""
    # 过滤空文本
    df = df[df[TEXT_COL].notna() & (df[TEXT_COL] != "")].reset_index(drop=True)

    # 逐行检测
    noise_results = []
    for idx, row in df.iterrows():
        is_noise, noise_type = detect_noise_sample(row)
        noise_results.append({
            "样本索引": idx,
            "文本": row[TEXT_COL],
            "标注标签": row[LABEL_COL],
            "情感得分": calculate_sentiment_score(row[TEXT_COL]),
            "是否噪声": is_noise,
            "噪声类型": noise_type
        })

    noise_df = pd.DataFrame(noise_results)
    return noise_df


# ====================== 3. 可视化统计 ======================
def print_noise_statistics(noise_df):
    """打印噪声统计信息"""
    total_samples = len(noise_df)
    noise_samples = noise_df[noise_df["是否噪声"] == True]
    noise_ratio = len(noise_samples) / total_samples * 100

    print("=" * 50)
    print("噪声样本检测结果统计")
    print("=" * 50)
    print(f"总样本数：{total_samples}")
    print(f"噪声样本数：{len(noise_samples)}")
    print(f"噪声比例：{noise_ratio:.2f}%")
    print("\n噪声类型分布：")
    print(noise_samples["噪声类型"].value_counts())
    print("\n前10条噪声样本示例：")
    print(noise_samples.head(10)[["文本", "标注标签", "噪声类型"]].to_string(index=False))


# ====================== 4. 主执行逻辑（一键运行） ======================
if __name__ == "__main__":
    # 加载数据集
    try:
        df = pd.read_csv(INPUT_CSV_PATH, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT_CSV_PATH, encoding="gbk")  # 兼容GBK编码

    # 检测噪声
    noise_df = batch_detect_noise(df)

    # 保存噪声样本
    noise_samples = noise_df[noise_df["是否噪声"] == True]
    noise_samples.to_csv(OUTPUT_NOISE_CSV, index=False, encoding="utf-8-sig")

    # 打印统计结果
    print_noise_statistics(noise_df)

    print(f"\n✅ 噪声样本已保存至：{OUTPUT_NOISE_CSV}")