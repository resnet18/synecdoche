import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path

try:
    import krippendorff
    HAS_KRIPP = True
except ImportError:
    HAS_KRIPP = False

def kendalls_w(ratings_df):
    n_items, m_raters = ratings_df.shape
    if m_raters < 2:
        return np.nan
    ranks = ratings_df.rank(method='average', axis=0)
    Ri = ranks.sum(axis=1)
    R_mean = Ri.mean()
    numerator = 12 * ((Ri - R_mean) ** 2).sum()
    denominator = (m_raters ** 2) * (n_items ** 3 - n_items)
    return numerator / denominator

# 路径配置
base_dir = Path("D:/Projects/synecdoche")
input_file = base_dir / "data/evaluation/human_scores_cleaned.csv"
output_dir = base_dir / "data/evaluation"

df = pd.read_csv(input_file)
score_cols = [c for c in df.columns if c not in ['text_id', 'text']]

groups = {
    'Players': ['a_priori_0', 'a_priori_1', 'a_priori_2', 'a_priori_3'],
    'Naïve':   ['a_posteriori_0', 'a_posteriori_1', 'a_posteriori_2'],
    'Expert':  ['expertised_0'],
    'Researcher': ['first_author_0']
}

# 原始列名引用（修复bug）
researcher_col = groups['Researcher'][0]

# 计算各组平均数
group_means = {}
for gname, cols in groups.items():
    group_means[gname] = df[cols].mean(axis=1)

print("=" * 70)
print("结果分析")
print("=" * 70)
print(f"\n【样本信息】")
print(f"测试集: {len(df)}句 × {len(score_cols)}标注者")

# === 1. 组内一致性 ===
print(f"\n{'='*70}")
print("【组内一致性 (Intra-group Reliability)】")
print(f"{'='*70}")

for gname, cols in groups.items():
    subset = df[cols]
    n_raters = len(cols)
    print(f"\n{gname}: n={n_raters}")
    
    if n_raters < 2:
        print("  n<2, 无法计算一致性")
        continue
    
    W = kendalls_w(subset)
    print(f"  Kendall's W (排序一致性): {W:.4f}", end="")
    if W > 0.7:
        print(" ✓✓ 优秀")
    elif W > 0.5:
        print(" ✓ 良好")
    elif W > 0.3:
        print(" △ 中等")
    else:
        print(" ⚠️ 较差")
    
    if HAS_KRIPP:
        alpha = krippendorff.alpha(subset.T.values, level_of_measurement='interval')
        print(f"  Krippendorff α (绝对值一致性): {alpha:.4f} (参考)")

# === 2. 组间一致性 ===
print(f"\n{'='*70}")
print("【组间对齐度 (Inter-group Spearman ρ)】")
print("基于各组平均数作为群体共识")
print(f"{'='*70}")

group_names = list(groups.keys())
for i in range(len(group_names)):
    for j in range(i+1, len(group_names)):
        g1, g2 = group_names[i], group_names[j]
        m1, m2 = group_means[g1], group_means[g2]
        mask = m1.notna() & m2.notna()
        if mask.sum() > 10:
            rho, p = spearmanr(m1[mask], m2[mask])
            print(f"  {g1} vs {g2}: ρ={rho:.4f} (p={p:.4e})")

# === 3. 分歧句分析 ===
print(f"\n{'='*70}")
print("【分歧句分析 (Researcher vs Players Consensus)】")
print(f"{'='*70}")

players_mean = group_means['Players']
researcher_mean = group_means['Researcher']
df['players_mean'] = players_mean
df['diff'] = researcher_mean - players_mean
df['diff_abs'] = np.abs(df['diff'])

top_diff = df.nlargest(10, 'diff_abs')
print("\nTop 10 分歧最大的句子 (|Researcher - Players_Mean|):")
for _, row in top_diff.iterrows():
    direction = "你偏高" if row['diff'] > 0 else "你偏低"
    # 修复：用原始列名 researcher_col 而不是 'Researcher'
    print(f"\n  [ID:{row['text_id']}] 差异={row['diff_abs']:.1f}分 ({direction})")
    print(f"    文本: {row['text'][:60]}...")
    print(f"    你:{row[researcher_col]:.1f} | 玩家均值:{row['players_mean']:.1f}")

top_diff.to_csv(output_dir / "divergent_sentences_researcher_vs_players.csv", 
                index=False, encoding='utf-8-sig')
print(f"\n✓ 分歧句列表已保存")

# === 4. 摘要 ===
print(f"\n{'='*70}")
print("【关键发现摘要】")
print(f"{'='*70}")
for gname, cols in groups.items():
    if len(cols) >= 2:
        print(f"{gname}组 Kendall's W: {kendalls_w(df[cols]):.3f}")

mask = group_means['Players'].notna() & group_means['Researcher'].notna()
rho_you, _ = spearmanr(group_means['Players'][mask], group_means['Researcher'][mask])
print(f"Players vs Researcher: Spearman ρ = {rho_you:.3f}")
print(f"最大分歧幅度: {df['diff_abs'].max():.1f}分")
print(f"{'='*70}")