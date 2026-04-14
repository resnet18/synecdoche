import pandas as pd
from pathlib import Path

base_dir = Path("D:/Projects/synecdoche")

# 读取清洗后的人类分数
human_file = base_dir / "data/evaluation/human_scores_cleaned.csv"
df = pd.read_csv(human_file)

# 计算各组聚合分数
players_cols = ['a_priori_0', 'a_priori_1', 'a_priori_2', 'a_priori_3']
naive_cols = ['a_posteriori_0', 'a_posteriori_1']

df['score_players'] = df[players_cols].median(axis=1)  # 主基准：Players中位数
df['score_naive'] = df[naive_cols].median(axis=1)     # 路人中位数
df['score_expert'] = df['expertised_0']               # NLPer
df['score_researcher'] = df['first_author_0']         # 研究者（你）

# 创建输出目录
human_results_dir = base_dir / "results/final_validation/human"
human_results_dir.mkdir(parents=True, exist_ok=True)

# 导出4个人类基准文件（模仿机器预测格式：text_id, text, score）
outputs = {
    'predictions.csv': ('score_players', 'Players Median (Community Consensus)'),
    'predictions_naive.csv': ('score_naive', 'Naive Raters Median'),
    'predictions_expert.csv': ('score_expert', 'Expert Rater (NLP Background)'),
    'predictions_firstauthor.csv': ('score_researcher', 'Researcher (Self-Rating)')
}

for filename, (score_col, desc) in outputs.items():
    output_df = df[['text_id', 'text', score_col]].copy()
    output_df.columns = ['text_id', 'text', 'score']  # 统一列名
    output_file = human_results_dir / filename
    output_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✓ {filename}: {desc}, {len(output_df)}句, 保存到 {output_file}")

print(f"\n人类基准文件已准备就绪，路径: {human_results_dir}")
print("文件列表:")
for f in human_results_dir.glob("*.csv"):
    print(f"  - {f.name}")