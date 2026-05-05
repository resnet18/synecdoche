import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path

base_dir = Path("D:/Projects/synecdoche")

# ==================== 统一基线池（人类 + 机器平权） ====================
all_methods = {
    # 人类基线
    'H0': base_dir / "results/final_validation/human/predictions.csv",
    'H1': base_dir / "results/final_validation/human/predictions_naive.csv",
    'H2': base_dir / "results/final_validation/human/predictions_expert.csv",
    'H3': base_dir / "results/final_validation/human/predictions_firstauthor.csv",
    # 机器方法
    'B0':  base_dir / "results/final_validation/b0/predictions.csv",
    'B1':  base_dir / "results/final_validation/b1/predictions.csv",
    'B2':  base_dir / "results/final_validation/b2/predictions.csv",
    'B3':  base_dir / "results/final_validation/b3/predictions.csv",
    'B4':  base_dir / "results/final_validation/b4/predictions.csv",
    'B4p': base_dir / "results/final_validation/b4_pretrained/predictions.csv",
    'B5':  base_dir / "results/final_validation/b5/predictions.csv",
    'B6':  base_dir / "results/final_validation/b6/predictions.csv",
    'B6f': base_dir / "results/final_validation/b6_frozen/predictions.csv",
    'B7f': base_dir / "results/final_validation/b7_fp16/predictions.csv",
    'B7i': base_dir / "results/final_validation/b7_int4/predictions.csv",
    'B8':  base_dir / "results/final_validation/b8/predictions.csv",
    'B9':  base_dir / "results/final_validation/b9/predictions.csv",
    'O1':  base_dir / "results/final_validation/o1/predictions.csv",
    'O2':  base_dir / "results/final_validation/o2/predictions.csv",
    'O2p': base_dir / "results/final_validation/o2_probe/predictions.csv",
    'O3':  base_dir / "results/final_validation/o3/predictions.csv",
}

# 解码表（输出全称用）
decode = {
    'H0': 'Players', 'H1': 'Naive', 'H2': 'Expert', 'H3': 'Researcher',
    'B0': 'B0', 'B1': 'B1', 'B2': 'B2', 'B3': 'B3', 'B4': 'B4',
    'B4p': 'B4_pretrained', 'B5': 'B5', 'B6': 'B6', 'B6f': 'B6_frozen',
    'B7f': 'B7_fp16', 'B7i': 'B7_int4', 'B8': 'B8',
    'O1': 'O1', 'O2': 'O2', 'O2p': 'O2_probe', 'O3': 'O3'
}

# ==================== 函数层 ====================
def load_scores(path):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    score_col = next((c for c in df.columns if 'score' in c), None)
    if score_col is None:
        raise ValueError(f"{path.name}: 无score列")
    return df[['text_id', score_col]].rename(columns={score_col: 'score'})

# ==================== 加载 ====================
preds = {}
for key, path in all_methods.items():
    df = load_scores(path)
    if df is not None:
        preds[key] = df.set_index('text_id')['score'].sort_index()

if len(preds) < 2:
    raise SystemExit("有效方法不足2个")

df_wide = pd.DataFrame(preds)
names = list(df_wide.columns)
n = len(names)

# ==================== 计算全矩阵 ====================
metrics = {'Spearman': np.eye(n), 'MSE': np.zeros((n, n)), 
           'MAE': np.zeros((n, n)), 'Bias': np.zeros((n, n))}

for i, a in enumerate(names):
    for j, b in enumerate(names):
        if i == j:
            metrics['MSE'][i,j] = 0.0
            metrics['MAE'][i,j] = 0.0
            metrics['Bias'][i,j] = 0.0
            continue
        
        valid = df_wide[[a, b]].dropna()
        if len(valid) == 0:
            for k in metrics:
                metrics[k][i,j] = np.nan
            continue
        
        x, y = valid[a], valid[b]
        metrics['Spearman'][i,j], _ = spearmanr(x, y)
        metrics['MSE'][i,j] = np.mean((x - y)**2)
        metrics['MAE'][i,j] = np.mean(np.abs(x - y))
        metrics['Bias'][i,j] = x.mean() - y.mean()

matrices = {k: pd.DataFrame(v, index=names, columns=names) for k, v in metrics.items()}

# ==================== 输出 ====================
print("=" * 80)
print("ALL-METHOD ALIGNMENT (Humans as H0-H3, Machines as B*/O*)")
print("=" * 80)

for metric_name, mat in matrices.items():
    print(f"\n【{metric_name}】")
    print(mat.round(3 if metric_name == 'Spearman' else 2))

# 长格式配对
pairs = []
for i in range(n):
    for j in range(i+1, n):
        a, b = names[i], names[j]
        pairs.append({
            'A': a, 'B': b,
            'A_full': decode.get(a, a), 'B_full': decode.get(b, b),
            'Spearman': metrics['Spearman'][i,j],
            'MSE': metrics['MSE'][i,j],
            'MAE': metrics['MAE'][i,j],
            'Bias': metrics['Bias'][i,j]
        })

pairs_df = pd.DataFrame(pairs)

print("\n" + "=" * 80)
print("TOP 5 最接近 (最小 MSE)")
print("=" * 80)
print(pairs_df.nsmallest(5, 'MSE')[['A_full', 'B_full', 'Spearman', 'MSE', 'MAE', 'Bias']].to_string(index=False))

print("\n" + "=" * 80)
print("TOP 5 最偏离 (最大 MSE)")
print("=" * 80)
print(pairs_df.nlargest(5, 'MSE')[['A_full', 'B_full', 'Spearman', 'MSE', 'MAE', 'Bias']].to_string(index=False))

# ==================== 保存 ====================
out_dir = base_dir / "data/evaluation"
out_dir.mkdir(parents=True, exist_ok=True)

for metric_name, mat in matrices.items():
    mat.to_csv(out_dir / f"matrix_all_{metric_name.lower()}.csv")
pairs_df.to_csv(out_dir / "all_pairs.csv", index=False)

print(f"\n{'='*80}")
print("OUTPUT")
print(f"{'='*80}")
for k in matrices:
    print(f"  {out_dir / f'matrix_all_{k.lower()}.csv'}")
print(f"  {out_dir / 'all_pairs.csv'}")