import pandas as pd
import numpy as np
import re
import os

# === 路径配置 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'b2')
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'final', 'train.csv')
VALID_PATH = os.path.join(BASE_DIR, 'data', 'final', 'validation.csv')
TEST_PATH = os.path.join(BASE_DIR, 'data', 'test', 'test_positive.csv')

OUTPUT_PRED = os.path.join(RESULTS_DIR, 'predictions.csv')
OUTPUT_CENTER = os.path.join(RESULTS_DIR, 'center_vector.npy')

print("=== B2 标点指纹基线 ===")

# 读取数据
train_df = pd.read_csv(TRAIN_PATH)
valid_df = pd.read_csv(VALID_PATH)
test_df = pd.read_csv(TEST_PATH)
train_texts = train_df['text'].astype(str).tolist()
valid_texts = valid_df['text'].astype(str).tolist()
all_texts = train_texts + valid_texts        # 合并
test_texts = test_df['text'].astype(str).tolist()

def extract_punctuation_features(text):
    """提取标点特征：省略号、破折号、逗号、句号密度（占总字数比例）"""
    # === 关键修复：去除脱敏占位符 [概念] [属性] 等 ===
    clean_text = re.sub(r'\[[^\]]+\]', '', text)  # 去除方括号及其内容
    
    length = max(len(clean_text), 1)  # 基于清理后的文本长度
    
    # 基于clean_text统计各类标点（不再统计方括号）
    ellipsis = len(re.findall(r'…+|\.{3,}', clean_text))
    dash = len(re.findall(r'——|--+', clean_text))
    comma = len(re.findall(r'，|,', clean_text))
    period = len(re.findall(r'。|\.', clean_text))
    question = len(re.findall(r'？|\?', clean_text))
    exclaim = len(re.findall(r'！|!', clean_text))
    
    # 返回密度特征（与长度无关）
    return np.array([
        ellipsis / length,
        dash / length,
        comma / length,
        period / length,
        question / length,
        exclaim / length
    ])

# 训练集+验证集特征提取
print("提取训练集和验证集的标点特征...")
all_features = np.array([extract_punctuation_features(t) for t in all_texts])
center_vector = np.mean(all_features, axis=0).reshape(1, -1)
print(f"标点特征维度：6维（省略号/破折号/逗号/句号/问号/感叹号密度）")
print(f"中心向量（训练集平均）：{center_vector[0]}")

# 保存中心向量
np.save(OUTPUT_CENTER, center_vector)

# 测试集特征提取
print("提取测试集标点特征...")
test_features = np.array([extract_punctuation_features(t) for t in test_texts])

# 计算欧氏距离（距离越小越相似），转成0-100分
from sklearn.metrics.pairwise import euclidean_distances
distances = euclidean_distances(test_features, center_vector).flatten()
# 距离转相似度：用负指数或简单归一化，这里用(1 - 归一化距离) * 100
max_dist = distances.max() if distances.max() > 0 else 1
scores = (1 - distances / max_dist) * 100

# 保存结果
results = pd.DataFrame({
    'text': test_texts,
    'b2_score': scores,
    'distance': distances
})
results.to_csv(OUTPUT_PRED, index=False, encoding='utf-8-sig')

print(f"\n结果保存至：{OUTPUT_PRED}")
print(f"分数统计：min={scores.min():.2f}, max={scores.max():.2f}, mean={scores.mean():.2f}")