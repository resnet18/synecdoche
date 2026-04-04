import pandas as pd
import numpy as np
import re
import os
from sklearn.metrics.pairwise import euclidean_distances

# === 路径配置（4层dirname适配final_validation路径） ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 输入：直接调用预实验计算好的中心向量（只读）
PRETRAIN_DIR = os.path.join(BASE_DIR, 'results', 'b2')
CENTER_PATH = os.path.join(PRETRAIN_DIR, 'center_vector.npy')

# 输入：最终测试集（210句完整测试集）
TEST_PATH = os.path.join(BASE_DIR, 'data', 'test', 'test_full.csv')

# 输出：最终验证结果目录
OUTPUT_DIR = os.path.join(BASE_DIR, 'results', 'final_validation', 'b2')
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PRED = os.path.join(OUTPUT_DIR, 'predictions.csv')

print("=== B2 标点指纹基线 - 最终测试集推理 ===")

def extract_punctuation_features(text):
    """提取标点特征：省略号、破折号、逗号、句号密度（占总字数比例）"""
    # 去除脱敏占位符 [概念] [属性] 等
    clean_text = re.sub(r'\[[^\]]+\]', '', text)
    length = max(len(clean_text), 1)
    
    # 基于clean_text统计各类标点
    ellipsis = len(re.findall(r'…+|\.{3,}', clean_text))
    dash = len(re.findall(r'——|--+', clean_text))
    comma = len(re.findall(r'，|,', clean_text))
    period = len(re.findall(r'。|\.', clean_text))
    question = len(re.findall(r'？|\?', clean_text))
    exclaim = len(re.findall(r'！|!', clean_text))
    
    # 返回密度特征（与长度无关的稳健特征）
    return np.array([
        ellipsis / length,
        dash / length,
        comma / length,
        period / length,
        question / length,
        exclaim / length
    ])

# 1. 加载预计算的风格锚点（中心向量）
print(f"加载预计算中心向量: {CENTER_PATH}")
center_vector = np.load(CENTER_PATH).reshape(1, -1)
print(f"标点特征维度: 6维（省略号/破折号/逗号/句号/问号/感叹号密度）")
print(f"锚点向量: {center_vector[0].round(4)}")

# 2. 读取最终测试集（210句，含ID/OOD）
test_df = pd.read_csv(TEST_PATH)
test_texts = test_df['text'].astype(str).tolist()
print(f"测试集规模: {len(test_texts)}句")

# 3. 提取测试集特征并计算偏离度
print("提取测试集标点特征...")
test_features = np.array([extract_punctuation_features(t) for t in test_texts])

# 计算欧氏距离（距离越小风格越接近锚点）
distances = euclidean_distances(test_features, center_vector).flatten()

# 距离转相似度分数（0-10分，与项目其他基线保持一致）
# 使用负指数归一化：score = 10 * exp(-distance * scale)
# 或使用简单线性映射（这里选择线性，保留距离原始分布供分析）
max_dist = distances.max() if distances.max() > 0 else 1
scores = (1 - distances / max_dist) * 10  # 映射到0-10分

# 4. 保存结果（等待人类基准对齐计算Spearman）
results = pd.DataFrame({
    'text_id': test_df.get('text_id', range(len(test_texts))),
    'text': test_texts,
    'b2_score': scores,
    'euclidean_distance': distances  # 保留原始距离供调试
})
results.to_csv(OUTPUT_PRED, index=False, encoding='utf-8-sig')

print(f"\n结果已保存: {OUTPUT_PRED}")
print(f"分数统计: min={scores.min():.2f}, max={scores.max():.2f}, mean={scores.mean():.2f}, std={scores.std():.2f}")
print(f"待人类基准打分完成后，合并计算Spearman相关性")

# ========== 显存报告（加在这里）==========
import torch
from pathlib import Path
import os

if torch.cuda.is_available():
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    # 从环境变量获取路径（父脚本传入），或默认当前目录
    peak_file = Path(os.environ.get('PEAK_FILE_PATH', '_gpu_peak_temp.txt'))
    peak_file.write_text(str(peak_mb))
    print(f"[GPU] 峰值显存: {peak_mb:.0f} MB")
else:
    # 即使没有GPU，也写入0，避免父脚本读取失败
    peak_file = Path(os.environ.get('PEAK_FILE_PATH', '_gpu_peak_temp.txt'))
    peak_file.write_text("0")