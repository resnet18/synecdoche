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

human_baselines = {
    'Players': base_dir / "results/final_validation/human/predictions.csv",
    'Naive': base_dir / "results/final_validation/human/predictions_naive.csv",
    'Expert': base_dir / "results/final_validation/human/predictions_expert.csv",
    'Researcher': base_dir / "results/final_validation/human/predictions_firstauthor.csv",
}

print("=" * 80)
print("human-machine alignment (Spearman + MSE + MAE)")
print("=" * 80)

results = []

for method_name, method_path in methods.items():
    if not method_path.exists():
        print(f"✗ {method_name}: 文件不存在")
        continue
    
    try:
        method_df = pd.read_csv(method_path)
        score_col = [c for c in method_df.columns if 'score' in c][0]
        method_scores = method_df[['text_id', score_col]].copy()
        method_scores.columns = ['text_id', 'machine_score']
        
    except Exception as e:
        print(f"✗ {method_name}: 读取失败 - {e}")
        continue
    
    for human_name, human_path in human_baselines.items():
        try:
            human_df = pd.read_csv(human_path)
            merged = pd.merge(method_scores, human_df[['text_id', 'score']], 
                            on='text_id', suffixes=('_machine', '_human'))
            
            if len(merged) == 0:
                continue
            
            # Spearman (排序一致性)
            rho, pval = spearmanr(merged['machine_score'], merged['score'])
            
            # MSE & MAE (绝对偏离度)
            mse = np.mean((merged['machine_score'] - merged['score']) ** 2)
            mae = np.mean(np.abs(merged['machine_score'] - merged['score']))
            
            # 系统性偏差 (Bias)
            bias = merged['machine_score'].mean() - merged['score'].mean()
            
            results.append({
                'Method': method_name,
                'Human_Baseline': human_name,
                'Spearman_Rho': rho,
                'P_Value': pval,
                'MSE': mse,
                'MAE': mae,
                'Bias': bias,
                'N_Samples': len(merged),
                'Method_Mean': merged['machine_score'].mean(),
                'Human_Mean': merged['score'].mean()
            })
            
            print(f"✓ {method_name} vs {human_name}: ρ={rho:.3f}, MSE={mse:.2f}, MAE={mae:.2f}, Bias={bias:+.2f}")
            
        except Exception as e:
            print(f"✗ {method_name} vs {human_name}: {e}")

results_df = pd.DataFrame(results)

# 生成透视矩阵
pivot_rho = results_df.pivot(index='Method', columns='Human_Baseline', values='Spearman_Rho')
pivot_mse = results_df.pivot(index='Method', columns='Human_Baseline', values='MSE')
pivot_mae = results_df.pivot(index='Method', columns='Human_Baseline', values='MAE')
pivot_bias = results_df.pivot(index='Method', columns='Human_Baseline', values='Bias')

print("\n" + "=" * 80)
print("SPEARMAN ρ (排序一致性)")
print("=" * 80)
print(pivot_rho.round(3))

print("\n" + "=" * 80)
print("MSE (均方误差, 0-100尺度，越小越接近)")
print("=" * 80)
print(pivot_mse.round(2))

print("\n" + "=" * 80)
print("MAE (平均绝对误差, 0-10尺度，越小越接近)")
print("=" * 80)
print(pivot_mae.round(2))

print("\n" + "=" * 80)
print("BIAS (系统性偏差, +表示机器偏高，-表示机器偏低)")
print("=" * 80)
print(pivot_bias.round(2))

# 保存
output_dir = base_dir / "data/evaluation"
results_df.to_csv(output_dir / "alignment_full_metrics.csv", index=False)
pivot_rho.to_csv(output_dir / "matrix_spearman.csv")
pivot_mse.to_csv(output_dir / "matrix_mse.csv")
pivot_mae.to_csv(output_dir / "matrix_mae.csv")
pivot_bias.to_csv(output_dir / "matrix_bias.csv")

print(f"\n✓ 完整结果保存: {output_dir}")
print("  - alignment_full_metrics.csv (长格式)")
print("  - matrix_spearman/mse/mae/bias.csv (透视矩阵)")