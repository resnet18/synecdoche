import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle

# === 路径配置 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'b3')
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'final', 'train.csv')
VALID_PATH = os.path.join(BASE_DIR, 'data', 'final', 'validation.csv')
TEST_PATH = os.path.join(BASE_DIR, 'data', 'test', 'test_positive.csv')

OUTPUT_PRED = os.path.join(RESULTS_DIR, 'predictions.csv')
OUTPUT_MODEL = os.path.join(RESULTS_DIR, 'vectorizer.pkl')
OUTPUT_CENTER = os.path.join(RESULTS_DIR, 'center_vector.npy')

print("=== B3 TF-IDF基线（L2归一化，长度无关） ===")

# 读取数据
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
valid_df = pd.read_csv(VALID_PATH)
train_texts = train_df['text'].astype(str).tolist()
valid_texts = valid_df['text'].astype(str).tolist()
all_texts = train_texts + valid_texts        # 合并
test_texts = test_df['text'].astype(str).tolist()

print(f"训练集：{len(train_texts)}句")
print(f"测试集：{len(test_texts)}句")

# 构建TF-IDF模型
vectorizer = TfidfVectorizer(
    max_features=None,
    ngram_range=(1, 1),     # unigram，可以改(1,2)试unigram+bigram
    stop_words=None,        # 保留虚词
    min_df=2,
    norm='l2',              # L2归一化，消除长度影响
    use_idf=True
)

all_vectors = vectorizer.fit_transform(all_texts)
print(f"TF-IDF维度：{all_vectors.shape[1]}个特征词")

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
    'b3_score': scores,
    'similarity_raw': similarities
})
results.to_csv(OUTPUT_PRED, index=False, encoding='utf-8-sig')

print(f"\n结果保存至：{RESULTS_DIR}")
print(f"分数统计：min={scores.min():.2f}, max={scores.max():.2f}, mean={scores.mean():.2f}")

# 在你原来的代码后面加这行，看看实际词数
print(f"实际词表大小: {len(vectorizer.get_feature_names_out())}")