import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle

# 路径配置（从 src/04_baselines/b0.py 定位到根目录）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'b0')
os.makedirs(RESULTS_DIR, exist_ok=True)

# 只读路径（data/）
TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'final', 'train.csv')
VALID_PATH = os.path.join(BASE_DIR, 'data', 'final', 'validation.csv')
TEST_PATH = os.path.join(BASE_DIR, 'data', 'test', 'test_positive.csv')

# 只写路径（results/b0/）
OUTPUT_PRED = os.path.join(RESULTS_DIR, 'predictions.csv')
OUTPUT_MODEL = os.path.join(RESULTS_DIR, 'vectorizer.pkl')
OUTPUT_CENTER = os.path.join(RESULTS_DIR, 'center_vector.npy')

# === 配置 ===
RANDOM_SEED = 325 # 已完成今日325大学习
MAX_FEATURES = 5000  # 限制维度

print("=== B0 词袋模型基线 ===")

# 1. 读取数据（只读，绝不写入data/）
train_df = pd.read_csv(TRAIN_PATH)
valid_df = pd.read_csv(VALID_PATH)
test_df = pd.read_csv(TEST_PATH)

train_texts = train_df['text'].astype(str).tolist()
valid_texts = valid_df['text'].astype(str).tolist()
all_texts = train_texts + valid_texts        # 合并
test_texts = test_df['text'].astype(str).tolist()

# 2. 构建词袋模型（无监督统计）
vectorizer = CountVectorizer(
    max_features=MAX_FEATURES,
    stop_words=None,        # 保留虚词（风格可能体现在"的""了"密度）
    ngram_range=(1, 1),     # 小样本+脱敏，用unigram更稳健
    min_df=2                # 至少出现2次
)

all_vectors = vectorizer.fit_transform(all_texts)
print(f"词袋维度：{all_vectors.shape[1]}个特征词")

# 3. 计算中心向量（训练集均值）
center_vector = np.array(all_vectors.mean(axis=0)).reshape(1, -1)
print("中心向量计算完成")

# 4. 测试集预测（只读test，结果写入results/）
test_vectors = vectorizer.transform(test_texts)
similarities = cosine_similarity(test_vectors, center_vector).flatten()
scores = similarities * 100  # 归一化0-100

# 5. 保存结果到 results/b0/（绝不污染data/）
# 保存模型和中心向量（下次直接加载，不用重新fit）
with open(OUTPUT_MODEL, 'wb') as f:
    pickle.dump(vectorizer, f)
np.save(OUTPUT_CENTER, center_vector)

# 保存预测分数（等人类打分后合并计算Spearman）
results = pd.DataFrame({
    'text': test_texts,
    'b0_score': scores,
    'similarity_raw': similarities
})
results.to_csv(OUTPUT_PRED, index=False, encoding='utf-8-sig')

print(f"\n结果已保存至：")
print(f"  模型：{OUTPUT_MODEL}")
print(f"  中心向量：{OUTPUT_CENTER}")
print(f"  预测分数：{OUTPUT_PRED}")
print(f"\n分数统计：min={scores.min():.2f}, max={scores.max():.2f}, mean={scores.mean():.2f}")