import os
import numpy as np
import pandas as pd
import requests
import gzip  # 新增：用于解压
from sklearn.metrics.pairwise import cosine_similarity
import jieba
from gensim.models import KeyedVectors

# === 路径配置（4层dirname适配final_validation路径） ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 输入：预实验计算好的中心向量（只读）
PRETRAIN_DIR = os.path.join(BASE_DIR, 'results', 'b4_pretrained')
CENTER_PATH = os.path.join(PRETRAIN_DIR, 'center_vector.npy')

# 模型缓存目录（FastText预训练权重）
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# 输入：最终测试集（210句完整测试集）
TEST_PATH = os.path.join(BASE_DIR, 'data', 'test', 'test_full.csv')

# 输出：最终验证结果目录
OUTPUT_DIR = os.path.join(BASE_DIR, 'results', 'final_validation', 'b4_pretrained')
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PRED = os.path.join(OUTPUT_DIR, 'predictions.csv')

# === 自动下载配置（FastText中文词向量） ===
PRETRAINED_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.vec.gz"
PRETRAINED_GZ = os.path.join(MODEL_DIR, "cc.zh.300.vec.gz")  # 压缩包
PRETRAINED_VEC = os.path.join(MODEL_DIR, "cc.zh.300.vec")    # 解压后的文本
PRETRAINED_BIN = os.path.join(MODEL_DIR, "cc.zh.300.bin")    # 二进制格式（目标）

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
            os.remove(dest_path)
        raise RuntimeError(f"下载失败: {e}\n请手动下载 {url} 并放置于 {dest_path}")

def ensure_binary_model():
    """
    确保二进制模型存在：
    1. 检查 .bin（最快加载）
    2. 没有则检查 .vec（转换一次）
    3. 没有则检查 .gz（解压）
    4. 没有则下载 -> 解压 -> 转换
    返回：.bin 文件路径
    """
    # 1. 目标已存在？直接返回
    if os.path.exists(PRETRAINED_BIN):
        return PRETRAINED_BIN
    
    # 2. 需要解压？
    if not os.path.exists(PRETRAINED_VEC) and os.path.exists(PRETRAINED_GZ):
        print("解压 .vec.gz （约6.6GB，需要一分钟）...")
        with gzip.open(PRETRAINED_GZ, 'rb') as f_in:
            with open(PRETRAINED_VEC, 'wb') as f_out:
                f_out.write(f_in.read())
        print(f"解压完成: {PRETRAINED_VEC}")
    
    # 3. 需要下载？
    if not os.path.exists(PRETRAINED_VEC):
        download_file(PRETRAINED_URL, PRETRAINED_GZ)
        # 下载完立刻解压
        print("解压中...")
        with gzip.open(PRETRAINED_GZ, 'rb') as f_in:
            with open(PRETRAINED_VEC, 'wb') as f_out:
                f_out.write(f_in.read())
        print(f"解压完成: {PRETRAINED_VEC}")
    
    # 4. 转换为二进制（首次运行耗时几分钟，但只执行一次）
    print("="*60)
    print("首次运行：将文本格式(.vec)转换为二进制格式(.bin)")
    print("加载速度将提升 10-50 倍，此过程需要 2-5 分钟...")
    print("="*60)
    
    # 加载文本格式（慢）
    model = KeyedVectors.load_word2vec_format(PRETRAINED_VEC, binary=False)
    
    # 保存为二进制（快）
    model.save_word2vec_format(PRETRAINED_BIN, binary=True)
    
    print(f"转换完成！二进制模型已保存: {PRETRAINED_BIN}")
    print(f"下次加载将秒开。")
    
    # 可选：删除 .vec 节省空间（保留 .gz 作为备份或也删除）
    # if os.path.exists(PRETRAINED_VEC):
    #     os.remove(PRETRAINED_VEC)
    #     print("已删除文本格式节省空间")
    
    return PRETRAINED_BIN

# === 主程序 ===
print("=== B4-Pretrained 预训练词向量基线 - 最终测试集推理 ===")

# 1. 确保模型就绪（二进制格式）
bin_path = ensure_binary_model()

# 2. 加载预计算的风格锚点（中心向量）
print(f"加载预计算中心向量: {CENTER_PATH}")
center_vector = np.load(CENTER_PATH).reshape(1, -1)
VECTOR_SIZE = center_vector.shape[1]
print(f"锚点向量维度: {VECTOR_SIZE}")

# 3. 读取最终测试集（210句，无需读取train/val）
test_df = pd.read_csv(TEST_PATH)
test_texts = test_df['text'].astype(str).tolist()
print(f"测试集规模: {len(test_texts)}句")

# 4. 加载预训练词向量模型（二进制格式，快）
print(f"加载FastText模型（二进制格式）: {bin_path}")
model = KeyedVectors.load_word2vec_format(bin_path, binary=True)  # binary=True 是关键
print(f"词向量维度: {model.vector_size}, 词表大小: {len(model.key_to_index):,}")

# 5. 分词与句子向量化
def tokenize(texts):
    return [jieba.lcut(text.strip()) for text in texts]

def sentence_vector(words, model, vector_size):
    vectors = [model[word] for word in words if word in model]
    if not vectors:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)

print("分词与向量化测试集...")
test_tokenized = tokenize(test_texts)
test_vectors = np.array([
    sentence_vector(words, model, VECTOR_SIZE) for words in test_tokenized
])

# 6. 计算与风格锚点的偏离度（0-10分）
print("计算风格偏离度...")
similarities = cosine_similarity(test_vectors, center_vector).flatten()
scores = similarities * 10  # 映射到0-10分，与其他基线一致

# 7. 保存结果（等待人类基准对齐）
results = pd.DataFrame({
    'text_id': test_df.get('text_id', range(len(test_texts))),
    'text': test_texts,
    'b4p_score': scores,
    'cosine_sim_raw': similarities
})
# 强制确保目录存在（双重保险）
os.makedirs(OUTPUT_DIR, exist_ok=True)
results.to_csv(OUTPUT_PRED, index=False, encoding='utf-8-sig')

print(f"\n结果已保存: {OUTPUT_PRED}")
print(f"分数统计: min={scores.min():.2f}, max={scores.max():.2f}, mean={scores.mean():.2f}, std={scores.std():.2f}")
print(f"待人类基准打分完成后，合并计算Spearman相关性")

# ========== 显存报告（加在这里）==========
import torch
from pathlib import Path

if torch.cuda.is_available():
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    peak_file = Path(os.environ.get('PEAK_FILE_PATH', '_gpu_peak_temp.txt'))
    peak_file.write_text(str(peak_mb))
    print(f"[GPU] 峰值显存: {peak_mb:.0f} MB")
else:
    # 即使没有GPU，也写入0，避免父脚本读取失败
    peak_file = Path(os.environ.get('PEAK_FILE_PATH', '_gpu_peak_temp.txt'))
    peak_file.write_text("0")