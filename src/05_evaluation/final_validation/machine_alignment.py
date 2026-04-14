import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path

base_dir = Path("D:/Projects/synecdoche")

methods = {
    'B0': base_dir / "results/final_validation/b0/predictions.csv",
    'B1': base_dir / "results/final_validation/b1/predictions.csv",
    'B2': base_dir / "results/final_validation/b2/predictions.csv",
    'B3': base_dir / "results/final_validation/b3/predictions.csv",
    'B4': base_dir / "results/final_validation/b4/predictions.csv",
    'B4_pretrained': base_dir / "results/final_validation/b4_pretrained/predictions.csv",
    'B5': base_dir / "results/final_validation/b5/predictions.csv",
    'B6': base_dir / "results/final_validation/b6/predictions.csv",
    'B6_frozen': base_dir / "results/final_validation/b6_frozen/predictions.csv",
    'B7_fp16': base_dir / "results/final_validation/b7_fp16/predictions.csv",
    'B7_int4': base_dir / "results/final_validation/b7_int4/predictions.csv",
    'B8': base_dir / "results/final_validation/b8/predictions.csv",
    'O1': base_dir / "results/final_validation/o1/predictions.csv",
    'O2': base_dir / "results/final_validation/o2/predictions.csv",
    'O2_probe': base_dir / "results/final_validation/o2_probe/predictions.csv",
    'O3': base_dir / "results/final_validation/o3/predictions.csv",
}

print("=" * 60)
print("inter-method reliability (Spearman + MSE + MAE)")
print("=" * 60)

all_preds = {}
for name, path in methods.items():
    if not path.exists():
        continue
    df = pd.read_csv(path)
    score_col = [c for c in df.columns if 'score' in c][0]
    all_preds[name] = df.set_index('text_id')[score_col].sort_index()

machine_df = pd.DataFrame(all_preds)

# 计算三种指标
n = len(machine_df.columns)
spearman_matrix = np.zeros((n, n))
mse_matrix = np.zeros((n, n))
mae_matrix = np.zeros((n, n))

for i, col_i in enumerate(machine_df.columns):
    for j, col_j in enumerate(machine_df.columns):
        if i == j:
            spearman_matrix[i, j] = 1.0
            mse_matrix[i, j] = 0.0
            mae_matrix[i, j] = 0.0
        else:
            # Spearman
            rho, _ = spearmanr(machine_df[col_i], machine_df[col_j])
            spearman_matrix[i, j] = rho
            
            # MSE & MAE
            mse = np.mean((machine_df[col_i] - machine_df[col_j]) ** 2)
            mae = np.mean(np.abs(machine_df[col_i] - machine_df[col_j]))
            mse_matrix[i, j] = mse
            mae_matrix[i, j] = mae

# 转为DataFrame
spearman_df = pd.DataFrame(spearman_matrix, index=machine_df.columns, columns=machine_df.columns)
mse_df = pd.DataFrame(mse_matrix, index=machine_df.columns, columns=machine_df.columns)
mae_df = pd.DataFrame(mae_matrix, index=machine_df.columns, columns=machine_df.columns)

print("\n【Spearman ρ (排序一致性)】")
print(spearman_df.round(3))

print("\n【MSE (均方误差)】")
print(mse_df.round(2))

print("\n【MAE (平均绝对误差)】")
print(mae_df.round(2))

# 找出偏离最大的配对
pairs = []
for i in range(n):
    for j in range(i+1, n):
        pairs.append({
            'Method_A': machine_df.columns[i],
            'Method_B': machine_df.columns[j],
            'Spearman': spearman_matrix[i, j],
            'MSE': mse_matrix[i, j],
            'MAE': mae_matrix[i, j]
        })

pairs_df = pd.DataFrame(pairs)

print("\n" + "=" * 60)
print("偏离最大的配对 (Top 5 MSE)")
print("=" * 60)
print(pairs_df.nlargest(5, 'MSE')[['Method_A', 'Method_B', 'Spearman', 'MSE', 'MAE']].to_string(index=False))

print("\n" + "=" * 60)
print("最接近的配对 (Top 5，最小MSE)")
print("=" * 60)
print(pairs_df.nsmallest(5, 'MSE')[['Method_A', 'Method_B', 'Spearman', 'MSE', 'MAE']].to_string(index=False))

# 保存
output_dir = base_dir / "data/evaluation"
spearman_df.to_csv(output_dir / "inter_method_spearman.csv")
mse_df.to_csv(output_dir / "inter_method_mse.csv")
mae_df.to_csv(output_dir / "inter_method_mae.csv")
pairs_df.to_csv(output_dir / "inter_method_pairs.csv", index=False)

print(f"\n✓ 结果保存到: {output_dir}")