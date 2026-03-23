import pandas as pd
import numpy as np
import jieba
import requests
from gensim.models import KeyedVectors
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import os

# ========== 路径配置 ===========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
BASE_DIR = os.path.join(PROJECT_ROOT, 'results', 'b4_pretrained')  # 指向预训练版结果目录
ABLATION_DIR = os.path.join(PROJECT_ROOT, 'results', 'ablation')
TEST_ORIG_PATH = os.path.join(PROJECT_ROOT, 'data', 'test', 'test_ablation.csv')
TRAIN_PATH = os.path.join(PROJECT_ROOT, 'data', 'final', 'train.csv')
VALID_PATH = os.path.join(PROJECT_ROOT, 'data', 'final', 'validation.csv')

os.makedirs(ABLATION_DIR, exist_ok=True)
os.makedirs(BASE_DIR, exist_ok=True)  # 确保目录存在

# ========== 自动下载配置（复用主实验逻辑） ===========
PRETRAINED_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.vec.gz"
PRETRAINED_FILENAME = "cc.zh.300.vec.gz"
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
PRETRAINED_PATH = os.path.join(MODEL_DIR, PRETRAINED_FILENAME)

def download_file(url, dest_path, chunk_size=8192*1024):
    if os.path.exists(dest_path):
        return
    print(f"首次运行：正在下载FastText中文词向量（1.2GB）...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r下载进度: {percent:.1f}%", end='', flush=True)
        print(f"\n下载完成: {dest_path}")
    except Exception as e:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise RuntimeError(f"下载失败: {e}")

# ========== 加载数据 ===========
print("=== B4-Pretrained 消融实验（通用词向量 vs 原文） ===")

# 自动下载（如缺失）
download_file(PRETRAINED_URL, PRETRAINED_PATH)

# 加载训练集（用于重建中心向量）
train_df = pd.read_csv(TRAIN_PATH)
valid_df = pd.read_csv(VALID_PATH)
train_texts = train_df['text'].astype(str).tolist()
valid_texts = valid_df['text'].astype(str).tolist()
all_train_texts = train_texts + valid_texts

# 加载原始测试集
df_test = pd.read_csv(TEST_ORIG_PATH)
test_texts = df_test['text'].astype(str).tolist()
print(f"加载训练语料: {len(all_train_texts)} 句 | 原始测试集: {len(test_texts)} 句")

# ========== 加载预训练模型并重建中心向量 ===========
print(f"加载预训练模型: {PRETRAINED_PATH}")
model = KeyedVectors.load_word2vec_format(PRETRAINED_PATH, binary=False)
vec_size = model.vector_size
print(f"词向量维度: {vec_size}")

def sentence_vector(words, model):
    vectors = [model[w] for w in words if w in model]
    if not vectors:
        return np.zeros(vec_size)
    return np.mean(vectors, axis=0)

# 重建中心向量（基于脱敏训练集，与主实验一致）
print("重建中心向量（基于脱敏训练集）...")
train_tokenized = [jieba.lcut(t.strip()) for t in all_train_texts]
train_vectors = np.array([sentence_vector(words, model) for words in train_tokenized])
center_vector = np.mean(train_vectors, axis=0).reshape(1, -1)
print(f"中心向量重建完成")

# ========== 计算原文分数 ===========
print("编码原文...")
test_tokenized = [jieba.lcut(t.strip()) for t in test_texts]
test_vectors = np.array([sentence_vector(words, model) for words in test_tokenized])
similarities = cosine_similarity(test_vectors, center_vector).flatten()
scores_orig = similarities  # 原始分数（0-1范围）

# 保存原文预测
df_orig = pd.DataFrame({
    'text': test_texts,
    'b4p_score': scores_orig * 100,  # b4p = b4-pretrained
    'similarity_raw': scores_orig
})
df_orig.to_csv(os.path.join(BASE_DIR, 'predictions_original.csv'), 
               index=False, encoding='utf-8-sig')
print(f"原文分数范围: [{scores_orig.min():.3f}, {scores_orig.max():.3f}]")

# ========== 加载脱敏分数并对比 ===========
des_path = os.path.join(BASE_DIR, 'predictions.csv')
if not os.path.exists(des_path):
    print(f"警告：找不到脱敏结果 {des_path}，请先运行主实验")
    exit()

df_des = pd.read_csv(des_path)
# 统一量纲（如果主实验存的是0-100，转回0-1）
des_scores = df_des['b4p_score'].values / 100 if df_des['b4p_score'].max() > 1 else df_des['b4p_score'].values

rho, p = spearmanr(des_scores, scores_orig)
print(f"\nSpearman ρ = {rho:.4f} (p={p:.4f})")

# ========== 更新汇总表（标记为B4P） ===========
summary_path = os.path.join(ABLATION_DIR, 'consistency_summary.csv')

new_record = {
    'method': 'B4P',  # 或 B4-Pretrained
    'spearman_rho': round(rho, 4),
    'p_value': round(p, 4)
}

try:
    if os.path.exists(summary_path):
        df_summary = pd.read_csv(summary_path)
        # 删除已有B4P记录
        df_summary = df_summary[df_summary['method'] != 'B4P']
    else:
        df_summary = pd.DataFrame(columns=['method', 'spearman_rho', 'p_value'])
    
    df_summary = pd.concat([df_summary, pd.DataFrame([new_record])], ignore_index=True)
    df_summary = df_summary.sort_values('method').reset_index(drop=True)
    df_summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"\n已更新汇总表: {summary_path}")
    print(df_summary)
except Exception as e:
    print(f"\n警告：无法更新汇总表: {e}")

# ========== 诊断输出 ===========
print(f"\n=== B4-Pretrained 消融诊断 ===")
print(f"原文OOV率估计: {np.mean([np.linalg.norm(v) < 1e-6 for v in test_vectors]):.1%}")
print(f"脱敏OOV率估计: 预计>50%（占位符全为OOV）")

if rho > 0.7:
    print("强一致：预训练词向量对专有词不敏感（意外）")
elif rho > 0.3:
    print("中等一致：部分词汇泄漏")
else:
    print("弱一致：严重词汇鸿沟（预期结果）")

print("=== 完成 ===")