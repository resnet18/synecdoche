import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle

# === 路径配置 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 输入：预实验训练好的模型（只读，绝不修改）
PRETRAIN_DIR = os.path.join(BASE_DIR, 'results', 'b0')
VECTORIZER_PATH = os.path.join(PRETRAIN_DIR, 'vectorizer.pkl')
CENTER_PATH = os.path.join(PRETRAIN_DIR, 'center_vector.npy')

# 输入：最终测试集
TEST_PATH = os.path.join(BASE_DIR, 'data', 'test', 'test_full.csv')

# 输出：最终验证结果（只写预测分数，不写模型）
OUTPUT_DIR = os.path.join(BASE_DIR, 'results', 'final_validation', 'b0')
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PRED = os.path.join(OUTPUT_DIR, 'predictions.csv')

print("=== B0 词袋基线 - 最终测试集推理（直接调用预实验模型） ===")

# 1. 加载预实验模型（零训练，纯推理）
print(f"加载预训练模型: {VECTORIZER_PATH}")
with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)
center_vector = np.load(CENTER_PATH)
print(f"词袋维度: {len(vectorizer.get_feature_names_out())}，中心向量形状: {center_vector.shape}")

# 2. 读取最终测试集（210句）
test_df = pd.read_csv(TEST_PATH)
test_texts = test_df['text'].astype(str).tolist()
print(f"测试集规模: {len(test_texts)}句")

# 3. 推理：计算与风格锚点的偏离度（0-10分）
test_vectors = vectorizer.transform(test_texts)  # 注意是 transform，不是 fit_transform
similarities = cosine_similarity(test_vectors, center_vector).flatten()
scores = similarities * 10  # 映射到 0-10 分

# 4. 保存结果（等待人类基准对齐）
results = pd.DataFrame({
    'text_id': test_df.get('text_id', range(len(test_texts))),
    'text': test_texts,
    'b0_score': scores,
    'cosine_sim_raw': similarities
})
results.to_csv(OUTPUT_PRED, index=False, encoding='utf-8-sig')

print(f"\n结果已保存: {OUTPUT_PRED}")
print(f"分数分布: min={scores.min():.2f}, max={scores.max():.2f}, mean={scores.mean():.2f}, std={scores.std():.2f}")
print(f"待人类基准完成后，运行 spearmanr 计算相关性")