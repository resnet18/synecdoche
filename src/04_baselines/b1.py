import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle

# === 路径配置 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'b1')
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'final', 'train.csv')
VALID_PATH = os.path.join(BASE_DIR, 'data', 'final', 'validation.csv')
TEST_PATH = os.path.join(BASE_DIR, 'data', 'test', 'test_positive.csv')

OUTPUT_PRED = os.path.join(RESULTS_DIR, 'predictions.csv')
OUTPUT_MODEL = os.path.join(RESULTS_DIR, 'vectorizer.pkl')
OUTPUT_CENTER = os.path.join(RESULTS_DIR, 'center_vector.npy')

# === 配置 ===
RANDOM_SEED = 325 # 已完成今日325大学习
MAX_FEATURES = 5000
NGRAM_RANGE = (2, 2)  # 只取bigram（2-gram），想看unigram+bigram就改(1,2)

print("=== B1 n-gram基线（bigram） ===")

# 读取数据
train_df = pd.read_csv(TRAIN_PATH)
valid_df = pd.read_csv(VALID_PATH)
test_df = pd.read_csv(TEST_PATH)
train_texts = train_df['text'].astype(str).tolist()
valid_texts = valid_df['text'].astype(str).tolist()
all_texts = train_texts + valid_texts        # 合并
test_texts = test_df['text'].astype(str).tolist()

# 构建n-gram模型
vectorizer = CountVectorizer(
    max_features=MAX_FEATURES,
    ngram_range=NGRAM_RANGE,    # bigram：捕获组合
    stop_words=None,
    min_df=2
)

all_vectors = vectorizer.fit_transform(all_texts)
print(f"n-gram维度：{all_vectors.shape[1]}个特征（范围{NGRAM_RANGE}）")

# 计算中心向量
center_vector = np.array(all_vectors.mean(axis=0)).reshape(1, -1)
print("中心向量计算完成")

# 测试集预测
test_vectors = vectorizer.transform(test_texts)
similarities = cosine_similarity(test_vectors, center_vector).flatten()
scores = similarities * 100

# 保存结果
with open(OUTPUT_MODEL, 'wb') as f:
    pickle.dump(vectorizer, f)
np.save(OUTPUT_CENTER, center_vector)

results = pd.DataFrame({
    'text': test_texts,
    'b1_score': scores,
    'similarity_raw': similarities
})
results.to_csv(OUTPUT_PRED, index=False, encoding='utf-8-sig')

print(f"\n结果保存至：{RESULTS_DIR}")
print(f"分数统计：min={scores.min():.2f}, max={scores.max():.2f}, mean={scores.mean():.2f}")