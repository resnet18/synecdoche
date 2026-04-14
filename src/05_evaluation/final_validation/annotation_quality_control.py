import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau
from pathlib import Path

try:
    import krippendorff
    HAS_KRIPP = True
except ImportError:
    HAS_KRIPP = False

def kendalls_w(ratings_df):
    """
    计算Kendall's W (Kendall's Coefficient of Concordance)
    ratings_df: DataFrame, shape=(n_items, m_raters)
    返回: W值 (0-1, 1=完全一致)
    """
    n_items, m_raters = ratings_df.shape
    if m_raters < 2:
        return np.nan
    
    # 计算每个item的秩和 (sum of ranks)
    # 先对每个rater的评分进行秩排序 (rank)，处理 ties 用 average
    ranks = ratings_df.rank(method='average', axis=0)
    Ri = ranks.sum(axis=1)  # 每行的秩和
    R_mean = Ri.mean()
    
    # Kendall's W 公式
    numerator = 12 * ((Ri - R_mean) ** 2).sum()
    denominator = (m_raters ** 2) * (n_items ** 3 - n_items)
    W = numerator / denominator
    
    return W

# 路径配置
base_dir = Path("D:/Projects/synecdoche")
input_file = base_dir / "data/evaluation/human_scores_cleaned.csv"
output_dir = base_dir / "data/evaluation"

df = pd.read_csv(input_file)
score_cols = [c for c in df.columns if c not in ['text_id', 'text']]
scores_df = df[score_cols]

# 分组定义
groups = {
    'Players (a_priori)': ['a_priori_0', 'a_priori_1', 'a_priori_2', 'a_priori_3'],
    'Naïve (a_posteriori)': ['a_posteriori_0', 'a_posteriori_1'],
    'Expert (expertised)': ['expertised_0'],
    'Researcher (first_author)': ['first_author_0']
}

print("=" * 70)
print("结果分析")
print("=" * 70)

print(f"\n【样本信息】")
print(f"测试集: {len(df)}句 × {len(score_cols)}标注者")

# === 1. 组内一致性：Kendall's W (排序) + Krippendorff (绝对值) ===
print(f"\n{'='*70}")
print("【组内一致性 (Intra-group Reliability)】")
print("注: Kendall's W 衡量排序一致性; Krippendorff's α 衡量绝对值一致性")
print(f"{'='*70}")

for group_name, cols in groups.items():
    subset = df[cols]
    n_raters = len(cols)
    print(f"\n{group_name}: n={n_raters}")
    
    if n_raters < 2:
        print("  n<2, 无法计算一致性")
        continue
    
    # Kendall's W (排序一致性 - 你的核心指标)
    W = kendalls_w(subset)
    print(f"  Kendall's W (排序一致性): {W:.4f}", end="")
    if W > 0.7:
        print(" ✓✓ 优秀 (排序高度一致)")
    elif W > 0.5:
        print(" ✓ 良好 (可接受)")
    elif W > 0.3:
        print(" △ 中等 (探索性)")
    else:
        print(" ⚠️ 较差 (随机性强)")
    
    # Krippendorff (绝对值一致性 - 参考)
    if HAS_KRIPP:
        alpha = krippendorff.alpha(subset.T.values, level_of_measurement='interval')
        print(f"  Krippendorff α (绝对值一致性): {alpha:.4f} (参考)")
    
    # 如果是n=2，额外显示两两相关
    if n_raters == 2:
        mask = subset.iloc[:, 0].notna() & subset.iloc[:, 1].notna()
        if mask.sum() > 10:
            rho, p = spearmanr(subset.iloc[:, 0][mask], subset.iloc[:, 1][mask])
            print(f"  两两Spearman: ρ={rho:.4f}")

# === 2. 分歧句分析 (Researcher vs Players) ===
# === 2. 分歧句分析 (Researcher vs Players) ===
print(f"\n{'='*70}")
print("【分歧句分析 (Researcher vs Community Consensus)】")
print("找出研究者(你)与玩家中位数评分差异最大的句子")
print(f"{'='*70}")

players = groups['Players (a_priori)']
researcher = groups['Researcher (first_author)'][0]

# 计算玩家中位数作为社区共识
players_median = df[players].median(axis=1)
researcher_score = df[researcher]

# 计算差异并添加到df（关键修正：明确赋值到df列）
df['players_median'] = players_median
df['diff_abs'] = np.abs(researcher_score - players_median)
df['diff'] = researcher_score - players_median  # 有方向：正=你偏高，负=你偏低

# Top10 分歧最大（列名修正）
columns_to_show = ['text_id', 'text', researcher] + players + ['diff_abs', 'diff', 'players_median']
top_diff = df.nlargest(10, 'diff_abs')[columns_to_show]

print("\nTop 10 分歧最大的句子 (|Researcher - Players_Median|):")
for idx, row in top_diff.iterrows():
    direction = "你偏高" if row['diff'] > 0 else "你偏低"
    players_scores_str = ", ".join([f"{row[p]:.0f}" for p in players])
    print(f"\n  [ID:{row['text_id']}] 差异={row['diff_abs']:.1f}分 ({direction})")
    print(f"    文本: {row['text'][:60]}...")
    print(f"    你:{row[researcher]:.0f} | 玩家中位数:{row['players_median']:.0f} | 玩家个体:[{players_scores_str}]")

# 保存分歧句
top_diff.to_csv(output_dir / "divergent_sentences_researcher_vs_players.csv", index=False, encoding='utf-8-sig')
print(f"\n✓ 完整分歧句列表已保存: divergent_sentences_researcher_vs_players.csv")

# === 3. 组间相关矩阵 (简要版) ===
print(f"\n{'='*70}")
print("【关键组间对齐度 (Spearman ρ)】")
print(f"{'='*70}")

# Players vs 其他
for group_name, cols in groups.items():
    if group_name == 'Players (a_priori)':
        continue
    for col in cols:
        mask = players_median.notna() & df[col].notna()
        if mask.sum() > 10:
            rho, p = spearmanr(players_median[mask], df[col][mask])
            print(f"  Players vs {col}: ρ={rho:.4f} (p={p:.4e})")

print(f"\n{'='*70}")
print("【关键发现摘要】")
print(f"{'='*70}")
print(f"1. Players组排序一致性: Kendall's W = {kendalls_w(df[players]):.3f}")
print(f"2. 你与社区共识的排序对齐: Spearman ρ = {spearmanr(players_median, researcher_score)[0]:.3f}")
print(f"3. 最大分歧句差异: {df['diff_abs'].max():.1f}分 (你vs玩家中位数)")
print(f"{'='*70}")