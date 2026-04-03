"""
O1: 纯文本结构指纹（零标签依赖版）- 最终测试集推理
基于 O1-v4：StandardScaler + 欧氏距离，映射到 0-10 分
"""

import pandas as pd
import numpy as np
import re
import jieba
import pickle
import os
from sklearn.metrics.pairwise import euclidean_distances

# ========== 路径配置（4层dirname适配final_validation） ==========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 输入：预实验训练好的 scaler 和中心向量（只读）
PRETRAIN_DIR = os.path.join(BASE_DIR, 'results', 'o1')
SCALER_PATH = os.path.join(PRETRAIN_DIR, 'scaler.pkl')
CENTER_PATH = os.path.join(PRETRAIN_DIR, 'center_vector.npy')

# 输入：最终测试集（210句完整测试集）
TEST_PATH = os.path.join(BASE_DIR, 'data', 'test', 'test_full.csv')

# 输出：final_validation标准路径
OUTPUT_DIR = os.path.join(BASE_DIR, 'results', 'final_validation', 'o1')
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PRED = os.path.join(OUTPUT_DIR, 'predictions.csv')

print("=== O1 结构指纹（零标签依赖版）- 最终测试集推理 ===")

# ========== 1. 语域词典（同预实验） ==========
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

# ========== 2. 特征提取（9维零标签依赖） ==========
def extract_o1_features(texts):
    """提取 O1-v4 的 9 维纯文本特征"""
    features = []
    for text in texts:
        if not isinstance(text, str) or not text:
            features.append(np.zeros(9))
            continue
        
        char_count = max(len(text), 1)
        words = list(jieba.cut(text))
        word_count = max(len(words), 1)
        
        sentences = [s.strip() for s in re.split(r'[。！？;；!?]', text) if s.strip()]
        sent_count = max(len(sentences), 1)
        
        # 语域标记密度
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
        
        features.append([
            func_density, literary_density, connector_density,
            avg_sent_len, clause_density, cv_sent_len,
            ellipsis_density, emotion_density, ultra_short_ratio
        ])
    
    return np.array(features)

# ========== 3. 加载预训练组件（零训练，纯推理） ==========
print(f"加载 StandardScaler: {SCALER_PATH}")
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

center_vector = np.load(CENTER_PATH)
print(f"加载中心向量: {center_vector.shape[0]}维")

# ========== 4. 读取最终测试集（210句） ==========
test_df = pd.read_csv(TEST_PATH)
test_texts = test_df['text'].astype(str).tolist()
print(f"测试集规模: {len(test_texts)}句")

# ========== 5. 特征提取与标准化 ==========
print("编码测试集...")
X_test = extract_o1_features(test_texts)
X_test_scaled = scaler.transform(X_test)  # 注意：transform，不是fit_transform

# ========== 6. 欧氏距离计算（与锚点的偏离度） ==========
distances = euclidean_distances(X_test_scaled, center_vector.reshape(1, -1)).flatten()

# 映射到 0-10 分（与项目其他基线保持一致）
# 距离越远 → 分数越低（越不典型）
max_dist = distances.max() if distances.max() > 0 else 1
scores = (1 - distances / max_dist) * 10

# ========== 7. 保存结果（包含std供分析） ==========
results = pd.DataFrame({
    'text_id': test_df.get('text_id', range(len(test_texts))),
    'text': test_texts,
    'o1_score': scores,              # 0-10分
    'euclidean_distance': distances   # 原始距离供调试
})
results.to_csv(OUTPUT_PRED, index=False, encoding='utf-8-sig')

# 打印统计（包含std，最终测试集独立格式）
print(f"\n结果已保存: {OUTPUT_PRED}")
print(f"分数统计: min={scores.min():.2f}, max={scores.max():.2f}, mean={scores.mean():.2f}, std={scores.std():.2f}")

# 分布诊断（std低说明所有样本挤在一起，std高说明区分度好）
if scores.std() < 1.0:
    print("警告: 标准差<1.0，分数分布过于集中，可能所有样本都被判为典型")
elif scores.std() > 3.0:
    print("良好: 标准差>3.0，分数分布分散，具备区分度")

print(f"\n待人类基准打分完成后，合并计算Spearman相关性")
print("=" * 60)