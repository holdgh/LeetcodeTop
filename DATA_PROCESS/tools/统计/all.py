import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# # 数据
# labels = ['异常', '正常', '通用', '噪声']
# sizes = [2145, 127, 52, 100]
# colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
# explode = (0.1, 0, 0, 0)  # 突出显示"异常"部分
#
# # 创建饼图
# plt.figure(figsize=(5, 4))
# plt.pie(sizes,
#         explode=explode,
#         labels=labels,
#         colors=colors,
#         autopct='%1.1f%%',
#         shadow=True,
#         startangle=140)
#
# # 添加标题
# plt.title('数据分布饼图', pad=20)
#
# # 确保图形是圆形
# plt.axis('equal')
#
# # 显示图例
# plt.legend(loc='upper right')
#
# # 显示图形
# plt.tight_layout()
# plt.show()


import matplotlib.pyplot as plt
import numpy as np

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 合并后的数据
defects = ['曲翘凸起', '刮刀横/竖条纹', '污染物', '球化', '铺粉不完全']
counts = [24357, 16694, 11533, 841, 63]
colors = ['#E74C3C', '#F39C12', '#3498DB', '#95A5A6', '#2ECC71']

# 创建图表
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制柱状图（自动按输入顺序排序）
bars = ax.bar(defects, counts, color=colors, width=0.6, edgecolor='white')

# 添加数据标签
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:,}',
            ha='center', va='bottom',
            fontsize=10)

# 特殊标记TOP1
ax.text(0, 24357*1.05, 'TOP1', ha='center', color='#E74C3C', fontweight='bold')

# 图表美化
ax.set_title('缺陷类型数量统计（按Box统计）', pad=20, fontsize=14)
ax.set_ylabel('数量', fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.4)

# Y轴使用科学计数法
ax.ticklabel_format(axis='y', style='plain')  # 普通数字格式
# 如果需要对数刻度可取消下方注释
# ax.set_yscale('log')

# # 添加统计说明
# plt.figtext(0.5, 0.01,
#             f"曲翘凸起占比 {24357/sum(counts):.1%} | 后三类缺陷合计占比 {(841+63+11533)/sum(counts):.1%}",
#             ha='center', fontsize=10)

plt.tight_layout()
plt.show()