"""
O1-v4: 结构特征（计算欧式距离，适用于程度打分）
"""

import pandas as pd
import numpy as np
import re
import os
import jieba
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import os

# ========== 1. 语域词典（同O1-v3） ==========
LITERARY_PARTICLES = {
    '之', '其', '而', '以', '于', '为', '与', '乃', '所', '被', '把', '且',
    '并', '却', '但', '然', '则', '故', '因', '虽', '若', '苟', '纵', '即',
    '已', '亦', '皆', '尽', '俱', '均', '咸', '悉', '皆尽', '全都',
    '尚', '仍', '还', '依然', '依旧', '仍然', '仍旧',
    '必', '定', '须', '应', '当', '该', '须得', '定要', '必将',
    '勿', '毋', '莫', '不', '未', '无', '非', '勿要', '不要', '不必',
    '甚', '颇', '极', '甚为', '很是', '相当', '非常',
    '何', '胡', '焉', '安', '孰', '谁', '何以', '为何',
    '即', '便', '就', '系', '是', '非', '即是', '便是', '就是',
    '仅', '只', '唯', '惟', '只是', '仅仅', '只不过'
}

MODAL_MARKERS = {
    '无妨', '不妨', '可否', '能否', '是否',
    '恐怕', '怕是', '唯恐', '生怕', '大概', '大约', '似乎', '好像',
    '应当', '应该', '理当', '该当', '自当', '本当',
    '未必', '不一定', '不见得', '难说', '未知', '未可知',
    '终究', '终将', '毕竟', '到底', '究竟', '终归', '总归', '迟早',
    '已然', '已经', '早已', '业已', '皆已', '俱已'
}

LOGICAL_CONNECTORS = {
    '因此', '所以', '因而', '从而', '于是', '因为', '由于', '缘于', '鉴于',
    '但是', '然而', '不过', '只是', '固然', '尽管', '虽然',
    '如果', '若', '若是', '假使', '假如', '倘若', '只要', '除非', '无论',
    '而且', '并且', '况且', '何况', '甚至', '乃至', '不仅', '不只', '不但'
}

FUNCTION_WORDS = LITERARY_PARTICLES | MODAL_MARKERS | LOGICAL_CONNECTORS

# ========== 2. 路径配置（4层dirname，预实验路径） ==========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'o1')
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'final', 'train.csv')
VALID_PATH = os.path.join(BASE_DIR, 'data', 'final', 'validation.csv')
TEST_PATH = os.path.join(BASE_DIR, 'data', 'test', 'test_positive.csv')

OUTPUT_PRED = os.path.join(RESULTS_DIR, 'predictions.csv')
OUTPUT_CENTER = os.path.join(RESULTS_DIR, 'center_vector.npy')
OUTPUT_SCALER = os.path.join(RESULTS_DIR, 'scaler.pkl')

print("=== O1-v4 欧氏距离版（真正程度计，非二分类器）===")

# ========== 3. 特征提取（同O1-v3，9维零标签依赖） ==========
def extract_textual_fingerprint(text):
    if not isinstance(text, str) or not text:
        return np.zeros(9)
    
    char_count = max(len(text), 1)
    words = list(jieba.cut(text))
    word_count = max(len(words), 1)
    
    sentences = [s.strip() for s in re.split(r'[。！？;；!?]', text) if s.strip()]
    sent_count = max(len(sentences), 1)
    
    # 语域标记
    func_count = sum(1 for w in words if w in FUNCTION_WORDS)
    func_density = func_count / word_count
    literary_count = sum(1 for w in words if w in LITERARY_PARTICLES)
    conn_count = sum(1 for w in words if w in LOGICAL_CONNECTORS)
    literary_density = literary_count / word_count
    connector_density = conn_count / word_count
    
    # 句法复杂度
    avg_sent_len = char_count / sent_count
    internal_pauses = len(re.findall(r'[，、；,;]', text))
    clause_density = internal_pauses / sent_count
    
    if len(sentences) > 1:
        sent_lens = [len(s) for s in sentences]
        cv_sent_len = np.std(sent_lens) / np.mean(sent_lens) if np.mean(sent_lens) > 0 else 0
    else:
        cv_sent_len = 0
    
    # 标点指纹
    ellipsis_count = len(re.findall(r'…+|\.{3,}', text))
    ellipsis_density = ellipsis_count / char_count
    emotion_count = text.count('？') + text.count('！') + text.count('?') + text.count('!')
    emotion_density = emotion_count / char_count
    ultra_short = sum(1 for s in sentences if len(s) <= 5)
    ultra_short_ratio = ultra_short / sent_count
    
    return np.array([
        func_density, literary_density, connector_density,
        avg_sent_len, clause_density, cv_sent_len,
        ellipsis_density, emotion_density, ultra_short_ratio
    ], dtype=np.float64)

# ========== 4. 主流程（欧氏距离版） ==========
# 读取数据
train_df = pd.read_csv(TRAIN_PATH)
valid_df = pd.read_csv(VALID_PATH)
test_df = pd.read_csv(TEST_PATH)

train_texts = train_df['text'].astype(str).tolist()
valid_texts = valid_df['text'].astype(str).tolist()
all_texts = train_texts + valid_texts
test_texts = test_df['text'].astype(str).tolist()

print(f"训练+验证集：{len(all_texts)}句")
print(f"测试集：{len(test_texts)}句")

# 提取原始特征（不做L2归一化）
print("提取特征...")
all_features = np.array([extract_textual_fingerprint(t) for t in all_texts])
test_features = np.array([extract_textual_fingerprint(t) for t in test_texts])

# StandardScaler（零均值单位方差，替代L2归一化）
print("StandardScaler标准化...")
scaler = StandardScaler()
all_features_scaled = scaler.fit_transform(all_features)
test_features_scaled = scaler.transform(test_features)

# 保存scaler（供消融实验和最终测试使用）
with open(OUTPUT_SCALER, 'wb') as f:
    pickle.dump(scaler, f)

# 中心向量（均值）
center_vector = np.mean(all_features_scaled, axis=0).reshape(1, -1)
np.save(OUTPUT_CENTER, center_vector)
print(f"中心向量（欧氏版）已保存: {center_vector.shape}")

# 欧氏距离计算（距离越小越典型）
print("计算欧氏距离...")
distances = euclidean_distances(test_features_scaled, center_vector).flatten()

# 距离转分数：0-100分制（与其他预实验基线一致）
max_dist = distances.max() if distances.max() > 0 else 1
scores = (1 - distances / max_dist) * 100  # 改回100，不是10

# 保存结果（列名保持o1_score，但现在是真正程度计）
results = pd.DataFrame({
    'text': test_texts,
    'o1_score': scores,
    'euclidean_distance': distances
})
results.to_csv(OUTPUT_PRED, index=False, encoding='utf-8-sig')

print(f"\n结果保存至：{OUTPUT_PRED}")
print(f"分数统计：min={scores.min():.2f}, max={scores.max():.2f}, mean={scores.mean():.2f}")

print("\n各特征均值（标准化前）：")
feature_names = [
    'func_density', 'literary_density', 'connector_density',
    'avg_sent_len', 'clause_density', 'cv_sent_len',
    'ellipsis_density', 'emotion_density', 'ultra_short_ratio'
]
for i, name in enumerate(feature_names):
    print(f"  {name}: {all_features[:, i].mean():.4f}")

print("\n=== O1-v4 完成 ===")
print("关键改进：")
print("  - 删除L2归一化（导致二分类的几何根源）")
print("  - 改用StandardScaler+欧氏距离（保留偏离度信息）")
print("  - 分数范围0-100分，距离越远分数越低（真正程度计）")
print("  - 原文/脱敏通用（零标签依赖特征保留）")