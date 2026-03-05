"""
O2 Ablation: SimCSE 原始 vs 脱敏一致性检验
与 B4/B5 保持相同的文件路径结构
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from scipy.stats import spearmanr
from torch.nn.functional import cosine_similarity

# ========== 路径配置（同 b4a.py 结构）==========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

BASE_DIR = os.path.join(PROJECT_ROOT, 'results', 'o2_probe')  # 原来是 'o2'
ABLATION_DIR = os.path.join(PROJECT_ROOT, 'results', 'ablation')  # 汇总表目录

# 输入文件（O2 训练产出）
MODEL_DIR = os.path.join(BASE_DIR, 'simcse_probe', 'final_model')
CENTER_PATH = os.path.join(BASE_DIR, 'center_vector.npy')

# 测试文件
TEST_ORIG_PATH = os.path.join(PROJECT_ROOT, 'data', 'test', 'test_positive.csv')
TEST_DESEN_PATH = os.path.join(PROJECT_ROOT, 'data', 'test', 'test_ablation.csv')

os.makedirs(ABLATION_DIR, exist_ok=True)

# ========== 加载模型和数据 ==========
print("=== O2 (SimCSE) 输入消融实验 ===\n")

df_orig = pd.read_csv(TEST_ORIG_PATH)
df_desen = pd.read_csv(TEST_DESEN_PATH)
texts_orig = df_orig['text'].astype(str).tolist()
texts_desen = df_desen['text'].astype(str).tolist()
print(f"加载测试集: {len(texts_orig)} 句")

model = SentenceTransformer(MODEL_DIR)
center_vector = np.load(CENTER_PATH)
center_tensor = torch.tensor(center_vector, dtype=torch.float32)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)  # 将整个模型（含projection）移到GPU/CPU
center_tensor = center_tensor.to(device)
print(f"使用设备: {device}")
print(f"SimCSE 模型加载完成，中心向量维度: {len(center_vector)}\n")

# ========== 评分函数 ==========
def predict_scores(texts):
    """使用投影头编码，输出128维"""
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            batch = texts[i:i+32]
            features = model.tokenize(batch)
            # 移到设备（device变量现在已全局定义）
            features = {k: v.to(device) for k, v in features.items()}
            
            # 前向传播：MiniLM(冻结) -> Projection
            outputs = model[0].auto_model(**features)
            pooler = outputs.pooler_output      # [batch, 384]
            features_dict = {'sentence_embedding': pooler}
            output_dict = model.projection(features_dict)
            proj = output_dict['sentence_embedding']
            embeddings.append(proj)
    
    embeddings = torch.cat(embeddings, dim=0)  # 已经在device上了
    # center_tensor 也已经在上面的新增代码中移到device了
    
    similarities = torch.nn.functional.cosine_similarity(
        embeddings, center_tensor.unsqueeze(0), dim=1
    )
    return (similarities.cpu().numpy() + 1) * 50  # 移回CPU转numpy
# ========== 计算两个版本 ==========
print("编码原始文本...")
scores_orig = predict_scores(texts_orig)

print("编码脱敏文本...")
scores_desen = predict_scores(texts_desen)

# 保存格式同 b4a.py：原始 predictions_original.csv，脱敏 predictions.csv
df_out_orig = pd.DataFrame({
    'text': texts_orig,
    'o2_score': scores_orig
})
df_out_orig.to_csv(os.path.join(BASE_DIR, 'predictions_original.csv'), 
                   index=False, encoding='utf-8-sig')

df_out_desen = pd.DataFrame({
    'text': texts_desen,
    'o2_score': scores_desen
})
df_out_desen.to_csv(os.path.join(BASE_DIR, 'predictions.csv'), 
                    index=False, encoding='utf-8-sig')

print(f"\n原始分数: [{scores_orig.min():.2f}, {scores_orig.max():.2f}], mean={scores_orig.mean():.2f}")
print(f"脱敏分数: [{scores_desen.min():.2f}, {scores_desen.max():.2f}], mean={scores_desen.mean():.2f}")

# ========== 统计检验（同 b4a.py 汇总逻辑）==========
rho, p = spearmanr(scores_orig, scores_desen)
print(f"\nSpearman ρ = {rho:.4f} (p={p:.4e})")

# ========== 追加到汇总表（同 b4a.py 完全一致）==========
summary_path = os.path.join(ABLATION_DIR, 'consistency_summary.csv')
new_record = {'method': 'O2-probe', 'spearman_rho': round(rho, 4), 'p_value': round(p, 4)}  # 原来是 'O2'

try:
    if os.path.exists(summary_path):
        df_summary = pd.read_csv(summary_path)
        # 去重：如果已有 O2 记录则替换
        df_summary = df_summary[df_summary['method'] != 'O2-probe']
    else:
        df_summary = pd.DataFrame(columns=['method', 'spearman_rho', 'p_value'])
    
    df_summary = pd.concat([df_summary, pd.DataFrame([new_record])], ignore_index=True)
    df_summary = df_summary.sort_values('method').reset_index(drop=True)
    df_summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    
    print(f"\n已更新汇总表: {summary_path}")
    print(df_summary.to_string(index=False))
    
except Exception as e:
    print(f"\n警告：无法更新汇总表: {e}")
    print(f"手动记录: O2, ρ={rho:.4f}, p={p:.4e}")

# ========== 诊断输出 ===========
print("\n一致性判定:")
if rho > 0.8 and p < 0.001:
    print("强一致：对专有词不敏感")
elif rho > 0.5:
    print("中等一致：部分依赖专有词")
else:
    print("弱一致：严重词汇泄漏")

print("\n=== O2 Ablation 完成 ===")