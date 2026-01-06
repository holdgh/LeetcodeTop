import pandas as pd

if __name__ == '__main__':
    # 加载噪声检测结果
    noise_df = pd.read_csv("../../jd_comment/noise_samples.csv", encoding="utf-8-sig")
    # 加载原始数据集
    df = pd.read_csv("../../jd_comment/dev.csv", encoding="utf-8")  # 替换为你的原始数据路径

    # 过滤规则：仅删除情感得分绝对值≥2的噪声样本（高置信度）
    high_confidence_noise = noise_df[(noise_df["是否噪声"] == True) & (abs(noise_df["情感得分"]) >= 2)]
    # 提取需要删除的样本索引
    noise_indices = high_confidence_noise["样本索引"].tolist()

    # 清洗数据集
    df_cleaned = df.drop(noise_indices).reset_index(drop=True)
    # 保存清洗后的数据集
    df_cleaned.to_csv("../../jd_comment/cleaned_dataset.csv", index=False, encoding="utf-8-sig")

    # 统计清洗后效果
    print(f"原始样本数：{len(df)}")
    print(f"清洗后样本数：{len(df_cleaned)}")
    print(f"清洗后噪声比例：{(len(noise_df) - len(high_confidence_noise)) / len(df_cleaned) * 100:.2f}%")
