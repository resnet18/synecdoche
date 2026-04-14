import pandas as pd
import re
from pathlib import Path

base_dir = Path("D:/Projects/synecdoche")
raw_file = base_dir / "data/raw/human_score_raw.xlsx"
eval_dir = base_dir / "data/evaluation"
eval_dir.mkdir(parents=True, exist_ok=True)

xl = pd.ExcelFile(raw_file, engine='openpyxl')

# 检查长度
print("=== 工作表长度检查 ===")
for sheet_name in xl.sheet_names:
    df = pd.read_excel(raw_file, sheet_name=sheet_name, header=None, engine='openpyxl')
    print(f"{sheet_name}: {len(df)}行 × {df.shape[1]}列")

# 取第一个工作表的句子列
first_df = pd.read_excel(raw_file, sheet_name=xl.sheet_names[0], header=None, engine='openpyxl')
sentences = first_df.iloc[:, 0].astype(str).str.replace(r'^\d+[、.．]\s*', '', regex=True).values[:210]

all_raters = {}

for sheet_name in xl.sheet_names:
    safe_name = sheet_name.replace(' ', '_')
    df = pd.read_excel(raw_file, sheet_name=sheet_name, header=None, engine='openpyxl')
    
    n_cols = df.shape[1]
    for i in range(1, n_cols):
        rater_name = f"{safe_name}_{i-1}"
        scores = pd.to_numeric(df.iloc[:, i], errors='coerce')
        all_raters[rater_name] = scores.values[:210]

# 构建DataFrame：text_id + text + 所有评分
human_scores = pd.DataFrame({
    'text_id': range(210),  # 0-209
    'text': sentences[:210],
    **all_raters
})

print(f"\n✓ 形状: {human_scores.shape}")
print(f"✓ 列: {list(human_scores.columns)[:5]}...")  # 显示前5列

# 保存
output_file = eval_dir / "human_scores_cleaned.csv"
human_scores.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n已保存: {output_file}")