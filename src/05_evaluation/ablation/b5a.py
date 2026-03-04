import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os

# ========== 路径 ===========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
BASE_DIR = os.path.join(PROJECT_ROOT, 'results', 'b5')
ABLATION_DIR = os.path.join(PROJECT_ROOT, 'results', 'ablation')
TEST_ORIG_PATH = os.path.join(PROJECT_ROOT, 'data', 'test', 'test_ablation.csv')

os.makedirs(ABLATION_DIR, exist_ok=True)

# ========== 加载 ===========
df_test = pd.read_csv(TEST_ORIG_PATH)
texts = df_test['text'].astype(str).tolist()
print(f"加载原始测试集: {len(texts)} 句")

# 加载MiniLM模型（自动使用GPU如果可用）
model = SentenceTransformer('all-MiniLM-L6-v2')
center_vector = np.load(os.path.join(BASE_DIR, 'center_vector.npy'))
print(f"中心向量维度: {center_vector.shape}")

# ========== 计算 ===========
print("编码原始文本（MiniLM）...")
# 直接编码整句，不需要jieba分词
vectors = model.encode(
    texts, 
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True  # 归一化，便于余弦计算
)

# 计算余弦相似度（向量已归一化，点积即余弦相似度）
scores = cosine_similarity(vectors, center_vector.reshape(1, -1)).flatten()

# 保存原始分数
df_orig = pd.DataFrame({
    'text': texts,
    'b5_score': scores * 100,
    'similarity_raw': scores
})
df_orig.to_csv(os.path.join(BASE_DIR, 'predictions_original.csv'), 
               index=False, encoding='utf-8-sig')
print(f"原始分数范围: [{scores.min():.3f}, {scores.max():.3f}]")

# ========== 对比脱敏版本 ===========
des_path = os.path.join(BASE_DIR, 'predictions.csv')
if not os.path.exists(des_path):
    print(f"错误：找不到脱敏结果 {des_path}")
    exit()

df_des = pd.read_csv(des_path)
des_scores = df_des['b5_score'].values / 100  # 统一量纲

rho, p = spearmanr(des_scores, scores)
print(f"\nSpearman ρ = {rho:.4f} (p={p:.4f})")

# ========== 追加到汇总表 ===========
summary_path = os.path.join(ABLATION_DIR, 'consistency_summary.csv')

new_record = {
    'method': 'B5',
    'spearman_rho': round(rho, 4),
    'p_value': round(p, 4)
}

try:
    if os.path.exists(summary_path):
        df_summary = pd.read_csv(summary_path)
        # 删除已有B5记录（避免重复）
        df_summary = df_summary[df_summary['method'] != 'B5']
    else:
        df_summary = pd.DataFrame(columns=['method', 'spearman_rho', 'p_value'])
    
    # 追加并排序
    df_summary = pd.concat([df_summary, pd.DataFrame([new_record])], ignore_index=True)
    df_summary = df_summary.sort_values('method').reset_index(drop=True)
    
    df_summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"\n已更新汇总表: {summary_path}")
    print(df_summary)
    
except Exception as e:
    print(f"\n警告：无法更新汇总表: {e}")
    print(f"手动记录: B5, ρ={rho:.4f}, p={p:.4f}")

# ========== 诊断 ===========
if rho > 0.7:
    print("\n强一致：B5对专有词不敏感（结构特征捕获成功）")
elif rho > 0.3:
    print("\n中等一致：部分依赖专有词")
else:
    print("\n弱一致：严重词汇泄漏")

print("=== B5消融完成 ===")