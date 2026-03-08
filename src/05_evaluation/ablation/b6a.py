"""
B6 Ablation: RoBERTa-large SimCSE 原始 vs 脱敏一致性检验
与 O2/B4/B5 保持相同的文件路径结构
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from scipy.stats import spearmanr
from torch.nn.functional import cosine_similarity

# ========== 路径配置（同 O2 结构）==========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

BASE_DIR = os.path.join(PROJECT_ROOT, 'results', 'b6')  # B6 结果根目录
ABLATION_DIR = os.path.join(PROJECT_ROOT, 'results', 'ablation')  # 汇总表目录

# 输入文件（B6 训练产出）
MODEL_DIR = os.path.join(BASE_DIR, 'roberta_simcse', 'final_model')
CENTER_PATH = os.path.join(BASE_DIR, 'center_vector.npy')

# 测试文件
TEST_ORIG_PATH = os.path.join(PROJECT_ROOT, 'data', 'test', 'test_ablation.csv')
TEST_DESEN_PATH = os.path.join(PROJECT_ROOT, 'data', 'test', 'test_positive.csv')

os.makedirs(ABLATION_DIR, exist_ok=True)

# ========== 加载模型和数据 ==========
print("=== B6 (RoBERTa-large SimCSE) 输入消融实验 ===\n")

df_orig = pd.read_csv(TEST_ORIG_PATH)
df_desen = pd.read_csv(TEST_DESEN_PATH)
texts_orig = df_orig['text'].astype(str).tolist()
texts_desen = df_desen['text'].astype(str).tolist()
print(f"加载测试集: {len(texts_orig)} 句")

model = SentenceTransformer(MODEL_DIR)
center_vector = np.load(CENTER_PATH)
center_tensor = torch.tensor(center_vector, dtype=torch.float32)
print(f"RoBERTa-large 模型加载完成，中心向量维度: {len(center_vector)}\n")

# ========== 评分函数（同 B5/O2 逻辑）==========
def predict_scores(texts):
    """计算与中心向量的余弦相似度，映射到 0-100"""
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
    center = center_tensor.to(embeddings.device)
    similarities = cosine_similarity(embeddings, center.unsqueeze(0), dim=1)
    return (similarities.cpu().numpy() + 1) * 50  # [-1,1] -> [0,100]

# ========== 计算两个版本 ==========
print("编码原始文本...")
scores_orig = predict_scores(texts_orig)

print("编码脱敏文本...")
scores_desen = predict_scores(texts_desen)

# 保存格式同 O2：原始 predictions_original.csv，脱敏 predictions.csv
df_out_orig = pd.DataFrame({
    'text': texts_orig,
    'b6_score': scores_orig
})
df_out_orig.to_csv(os.path.join(BASE_DIR, 'predictions_original.csv'), 
                   index=False, encoding='utf-8-sig')

df_out_desen = pd.DataFrame({
    'text': texts_desen,
    'b6_score': scores_desen
})
df_out_desen.to_csv(os.path.join(BASE_DIR, 'predictions.csv'), 
                    index=False, encoding='utf-8-sig')

print(f"\n原始分数: [{scores_orig.min():.2f}, {scores_orig.max():.2f}], mean={scores_orig.mean():.2f}")
print(f"脱敏分数: [{scores_desen.min():.2f}, {scores_desen.max():.2f}], mean={scores_desen.mean():.2f}")

# ========== 统计检验（同 O2 汇总逻辑）==========
rho, p = spearmanr(scores_orig, scores_desen)
print(f"\nSpearman ρ = {rho:.4f} (p={p:.4e})")

# ========== 追加到汇总表（同 O2 完全一致）==========
summary_path = os.path.join(ABLATION_DIR, 'consistency_summary.csv')
new_record = {'method': 'B6', 'spearman_rho': round(rho, 4), 'p_value': round(p, 4)}

try:
    if os.path.exists(summary_path):
        df_summary = pd.read_csv(summary_path)
        # 去重：如果已有 B6 记录则替换
        df_summary = df_summary[df_summary['method'] != 'B6']
    else:
        df_summary = pd.DataFrame(columns=['method', 'spearman_rho', 'p_value'])
    
    df_summary = pd.concat([df_summary, pd.DataFrame([new_record])], ignore_index=True)
    df_summary = df_summary.sort_values('method').reset_index(drop=True)
    df_summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    
    print(f"\n已更新汇总表: {summary_path}")
    print(df_summary.to_string(index=False))
    
except Exception as e:
    print(f"\n警告：无法更新汇总表: {e}")
    print(f"手动记录: B6, ρ={rho:.4f}, p={p:.4e}")

# ========== 诊断输出 ==========
print("\n一致性判定:")
if rho > 0.8 and p < 0.001:
    print("强一致：对专有词不敏感")
elif rho > 0.5:
    print("中等一致：部分依赖专有词")
else:
    print("弱一致：严重词汇泄漏")

print("\n=== B6 Ablation 完成 ===")