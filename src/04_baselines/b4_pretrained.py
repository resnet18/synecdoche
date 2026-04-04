import os
import numpy as np
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity
import jieba
from gensim.models import KeyedVectors

# === 路径配置 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'b4_pretrained')
os.makedirs(RESULTS_DIR, exist_ok=True)

# 模型缓存目录（自动创建）
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'final', 'train.csv')
VALID_PATH = os.path.join(BASE_DIR, 'data', 'final', 'validation.csv')
TEST_PATH = os.path.join(BASE_DIR, 'data', 'test', 'test_positive.csv')

OUTPUT_PRED = os.path.join(RESULTS_DIR, 'predictions.csv')
OUTPUT_CENTER = os.path.join(RESULTS_DIR, 'center_vector.npy')

# === 自动下载配置（FastText中文词向量） ===
PRETRAINED_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.vec.gz"
PRETRAINED_FILENAME = "cc.zh.300.vec.gz"
PRETRAINED_PATH = os.path.join(MODEL_DIR, PRETRAINED_FILENAME)
PRETRAINED_BIN = os.path.join(MODEL_DIR, "cc.zh.300.bin")  # 二进制缓存

def download_file(url, dest_path, chunk_size=8192*1024):
    """流式下载大文件，带进度显示"""
    if os.path.exists(dest_path):
        print(f"预训练模型已存在: {dest_path}")
        return
    
    print(f"正在下载FastText中文词向量（约1.2GB，首次运行需要几分钟）...")
    print(f"下载地址: {url}")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r下载进度: {percent:.1f}% ({downloaded/1024/1024:.1f}MB / {total_size/1024/1024:.1f}MB)", end='', flush=True)
        
        print(f"\n下载完成: {dest_path}")
        
    except Exception as e:
        if os.path.exists(dest_path):
            os.remove(dest_path)  # 清理半成品
        raise RuntimeError(f"下载失败: {e}\n请手动下载 {url} 并放置于 {dest_path}")

# === 主程序 ===
print("=== B4-Pretrained 预训练词向量基线（零训练，自动下载） ===")

# 1. 自动下载（如果缺失）
download_file(PRETRAINED_URL, PRETRAINED_PATH)

# 2. 读取数据（与B4相同）
train_df = pd.read_csv(TRAIN_PATH)
valid_df = pd.read_csv(VALID_PATH)
test_df = pd.read_csv(TEST_PATH)

train_texts = train_df['text'].astype(str).tolist()
valid_texts = valid_df['text'].astype(str).tolist()
all_texts = train_texts + valid_texts
test_texts = test_df['text'].astype(str).tolist()

print(f"\n训练语料（train+valid）: {len(all_texts)} 句")
print(f"测试集: {len(test_texts)} 句")

# 3. 分词
print("分词中...")
def tokenize(texts):
    return [jieba.lcut(text.strip()) for text in texts]

train_tokenized = tokenize(all_texts)
test_tokenized = tokenize(test_texts)

# 4. 加载预训练词向量（关键：直接读取.gz压缩文件，无需解压省空间）
# 确保二进制格式存在（首次转换后后续秒开）
if not os.path.exists(PRETRAINED_BIN):
    print("首次运行：转换文本格式为二进制（约2-5分钟，以后秒开）...")
    # 直接从.gz加载（兼容原逻辑）
    tmp_model = KeyedVectors.load_word2vec_format(PRETRAINED_PATH, binary=False)
    tmp_model.save_word2vec_format(PRETRAINED_BIN, binary=True)
    print(f"转换完成：{PRETRAINED_BIN}")
    del tmp_model  # 释放内存，下面重新加载二进制版
print(f"加载预训练词向量（可能需1-2分钟初始化）: {PRETRAINED_PATH}")
# 优先加载二进制（快），回退到.gz（兼容）
if os.path.exists(PRETRAINED_BIN):
    model = KeyedVectors.load_word2vec_format(PRETRAINED_BIN, binary=True)
    print(f"加载二进制模型（秒开）: {PRETRAINED_BIN}")
else:
    model = KeyedVectors.load_word2vec_format(PRETRAINED_PATH, binary=False)
    print(f"加载文本模型（慢）: {PRETRAINED_PATH}")

VECTOR_SIZE = model.vector_size
print(f"词向量维度: {VECTOR_SIZE}, 词表大小: {len(model.key_to_index):,}")

# 5. 句子向量函数（OOV跳过 + 平均池化）
def sentence_vector(words, model, vector_size):
    vectors = [model[word] for word in words if word in model]
    if not vectors:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)

# 6. 计算训练集中心向量
print("计算句子向量及中心向量...")
train_vectors = np.array([
    sentence_vector(words, model, VECTOR_SIZE) for words in train_tokenized
])
center_vector = np.mean(train_vectors, axis=0).reshape(1, -1)
print(f"中心向量计算完成，维度: {VECTOR_SIZE}")

# 7. 测试集预测
print("预测测试集...")
test_vectors = np.array([
    sentence_vector(words, model, VECTOR_SIZE) for words in test_tokenized
])
similarities = cosine_similarity(test_vectors, center_vector).flatten()
scores = similarities * 100

# 8. 保存结果
print("保存结果...")
np.save(OUTPUT_CENTER, center_vector)

results = pd.DataFrame({
    'text': test_texts,
    'b4p_score': scores,  # b4p = b4-pretrained
    'similarity_raw': similarities
})
results.to_csv(OUTPUT_PRED, index=False, encoding='utf-8-sig')

print(f"\n结果保存至：{RESULTS_DIR}")
print(f"分数统计：min={scores.min():.2f}, max={scores.max():.2f}, mean={scores.mean():.2f}")
print("=== B4-Pretrained 完成 ===")
print(f"\n注意：预训练模型文件（{PRETRAINED_FILENAME}）已缓存于 {MODEL_DIR}，后续运行将直接加载，无需重复下载")