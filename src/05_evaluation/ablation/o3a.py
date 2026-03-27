import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # 强制离线，不查更新
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = os.path.expanduser('~/.cache/huggingface')  # 确保缓存目录存在

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from transformers import ElectraTokenizer, ElectraModel
import torch

# ========== 路径 ===========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
BASE_DIR = os.path.join(PROJECT_ROOT, 'results', 'o3')  # 改为o3
ABLATION_DIR = os.path.join(PROJECT_ROOT, 'results', 'ablation')
TEST_ORIG_PATH = os.path.join(PROJECT_ROOT, 'data', 'test', 'test_ablation.csv')

os.makedirs(ABLATION_DIR, exist_ok=True)

# ========== 配置 ===========
MODEL_NAME = 'hfl/chinese-electra-180g-small-discriminator'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
MAX_LENGTH = 128

# ========== 加载 ===========
df_test = pd.read_csv(TEST_ORIG_PATH)
texts = df_test['text'].astype(str).tolist()
print(f"加载原始测试集: {len(texts)} 句")

# 加载ELECTRA-small模型（完全冻结）
print(f"加载模型: {MODEL_NAME}...")
tokenizer = ElectraTokenizer.from_pretrained(MODEL_NAME)
model = ElectraModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
for param in model.parameters():
    param.requires_grad = False

center_vector = np.load(os.path.join(BASE_DIR, 'center_vector.npy'))
print(f"中心向量维度: {center_vector.shape}")

# ========== 编码函数 ===========
@torch.no_grad()
def encode_texts(texts):
    """批量编码文本，返回 Mean Pooling 向量 (256维)"""
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True
        ).to(DEVICE)
        
        outputs = model(**inputs)
        # === 关键修改：Mean Pooling 替代 CLS ===
        mean_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        all_embeddings.append(mean_emb)
    
    return np.vstack(all_embeddings)

# ========== 计算 ===========
print("编码原始文本（ELECTRA-small）...")
vectors = encode_texts(texts)

# 归一化（L2 norm）以便余弦相似度计算 = 点积
vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
center_norm = center_vector / np.linalg.norm(center_vector)

# 计算余弦相似度（归一化后点积即余弦相似度）
scores = np.dot(vectors_norm, center_norm)

# 保存原始分数（映射到0-100，与主实验一致）
df_orig = pd.DataFrame({
    'text': texts,
    'o3_score': (scores + 1) / 2 * 100,  # [-1,1] -> [0,100]
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
# 脱敏分数转回[-1,1]范围以便比较
des_scores = df_des['o3_score'].values / 100 * 2 - 1

rho, p = spearmanr(des_scores, scores)
print(f"\nSpearman ρ = {rho:.4f} (p={p:.4f})")

# ========== 追加到汇总表 ===========
summary_path = os.path.join(ABLATION_DIR, 'consistency_summary.csv')

new_record = {
    'method': 'O3',
    'spearman_rho': round(rho, 4),
    'p_value': round(p, 4)
}

try:
    if os.path.exists(summary_path):
        df_summary = pd.read_csv(summary_path)
        # 删除已有O3记录（避免重复）
        df_summary = df_summary[df_summary['method'] != 'O3']
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
    print(f"手动记录: O3, ρ={rho:.4f}, p={p:.4f}")

# ========== 诊断 ===========
if rho > 0.7:
    print("\n强一致：O3对专有词不敏感")
elif rho > 0.3:
    print("\n中等一致：部分依赖专有词")
else:
    print("\n弱一致：严重词汇泄漏")

print("=== O3消融完成 ===")