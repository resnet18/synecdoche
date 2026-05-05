import pandas as pd
from pathlib import Path

base_dir = Path("D:/Projects/synecdoche")

human_file = base_dir / "data/evaluation/human_scores_cleaned.csv"
df = pd.read_csv(human_file)

# 群体列定义
players_cols = ['a_priori_0', 'a_priori_1', 'a_priori_2', 'a_priori_3']
naive_cols = ['a_posteriori_0', 'a_posteriori_1', 'a_posteriori_2']  # 补全3人

# 统一用平均数作为群体共识
df['score_players'] = df[players_cols].mean(axis=1)
df['score_naive'] = df[naive_cols].mean(axis=1)
df['score_expert'] = df['expertised_0']
df['score_researcher'] = df['first_author_0']

human_results_dir = base_dir / "results/final_validation/human"
human_results_dir.mkdir(parents=True, exist_ok=True)

outputs = {
    'predictions.csv': ('score_players', 'Players Mean (Community Consensus)'),
    'predictions_naive.csv': ('score_naive', 'Naive Raters Mean'),
    'predictions_expert.csv': ('score_expert', 'Expert Rater (NLP Background)'),
    'predictions_firstauthor.csv': ('score_researcher', 'Researcher (Self-Rating)')
}

for filename, (score_col, desc) in outputs.items():
    output_df = df[['text_id', 'text', score_col]].copy()
    output_df.columns = ['text_id', 'text', 'score']
    output_file = human_results_dir / filename
    output_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✓ {filename}: {desc}, {len(output_df)}句, 保存到 {output_file}")

print(f"\n人类基准文件已准备就绪，路径: {human_results_dir}")
print("文件列表:")
for f in human_results_dir.glob("*.csv"):
    print(f"  - {f.name}")