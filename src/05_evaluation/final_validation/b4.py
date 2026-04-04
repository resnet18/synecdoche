import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import jieba
from gensim.models import Word2Vec

# === 路径配置（4层dirname适配final_validation路径） ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 输入：直接调用预实验训练好的模型和锚点（只读）
PRETRAIN_DIR = os.path.join(BASE_DIR, 'results', 'b4')
MODEL_PATH = os.path.join(PRETRAIN_DIR, 'word2vec.model')
CENTER_PATH = os.path.join(PRETRAIN_DIR, 'center_vector.npy')

# 输入：最终测试集（210句完整测试集）
TEST_PATH = os.path.join(BASE_DIR, 'data', 'test', 'test_full.csv')

# 输出：最终验证结果目录
OUTPUT_DIR = os.path.join(BASE_DIR, 'results', 'final_validation', 'b4')
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PRED = os.path.join(OUTPUT_DIR, 'predictions.csv')

print("=== B4 Word2Vec基线 - 最终测试集推理 ===")

# 1. 加载预实验模型和风格锚点（零训练，纯推理）
print(f"加载预训练Word2Vec模型: {MODEL_PATH}")
model = Word2Vec.load(MODEL_PATH)
center_vector = np.load(CENTER_PATH).reshape(1, -1)
VECTOR_SIZE = model.vector_size
print(f"词向量维度: {VECTOR_SIZE}, 词表大小: {len(model.wv.key_to_index)}")
print(f"锚点向量已加载: {CENTER_PATH}")

# 2. 读取最终测试集（210句，含ID/OOD）
test_df = pd.read_csv(TEST_PATH)
test_texts = test_df['text'].astype(str).tolist()
print(f"测试集规模: {len(test_texts)}句")

# 3. 分词（jieba处理测试集）
def tokenize(texts):
    return [jieba.lcut(text.strip()) for text in texts]

print("分词处理测试集...")
test_tokenized = tokenize(test_texts)

# 4. 句子向量化函数（平均词向量，OOV跳过）
def sentence_vector(words, model, vector_size):
    vectors = [model.wv[word] for word in words if word in model.wv]
    if not vectors:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)

# 5. 计算测试集句子向量及与锚点的偏离度（0-10分）
print("编码测试集并计算偏离度...")
test_vectors = np.array([
    sentence_vector(words, model, VECTOR_SIZE) for words in test_tokenized
])
similarities = cosine_similarity(test_vectors, center_vector).flatten()
scores = similarities * 10  # 映射到0-10分，与项目其他基线保持一致

# 6. 保存结果（等待人类基准对齐计算Spearman）
results = pd.DataFrame({
    'text_id': test_df.get('text_id', range(len(test_texts))),
    'text': test_texts,
    'b4_score': scores,
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