import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle

# === 路径配置（4层dirname适配final_validation路径） ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 输入：直接调用预实验训练好的模型（只读，绝不重新训练）
PRETRAIN_DIR = os.path.join(BASE_DIR, 'results', 'b1')
VECTORIZER_PATH = os.path.join(PRETRAIN_DIR, 'vectorizer.pkl')
CENTER_PATH = os.path.join(PRETRAIN_DIR, 'center_vector.npy')

# 输入：最终测试集（210句完整测试集）
TEST_PATH = os.path.join(BASE_DIR, 'data', 'test', 'test_full.csv')

# 输出：最终验证结果目录
OUTPUT_DIR = os.path.join(BASE_DIR, 'results', 'final_validation', 'b1')
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PRED = os.path.join(OUTPUT_DIR, 'predictions.csv')

print("=== B1 n-gram基线（bigram）- 最终测试集推理 ===")

# 1. 加载预实验模型（零训练，纯推理）
print(f"加载预训练模型: {VECTORIZER_PATH}")
with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)
center_vector = np.load(CENTER_PATH)
print(f"n-gram维度: {len(vectorizer.get_feature_names_out())}个特征")

# 2. 读取最终测试集（210句，含ID/OOD）
test_df = pd.read_csv(TEST_PATH)
test_texts = test_df['text'].astype(str).tolist()
print(f"测试集规模: {len(test_texts)}句")

# 3. 推理：计算与风格锚点的偏离度（0-10分）
# 注意：transform而非fit_transform，绝不重新训练
test_vectors = vectorizer.transform(test_texts)
similarities = cosine_similarity(test_vectors, center_vector).flatten()

# 统一评分尺度为0-10分（与B0及其他基线保持一致）
scores = similarities * 10

# 4. 保存结果（等待人类基准对齐计算Spearman）
results = pd.DataFrame({
    'text_id': test_df.get('text_id', range(len(test_texts))),
    'text': test_texts,
    'b1_score': scores,
    'cosine_sim_raw': similarities
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