"""
O2 推理: 计算句子与中心向量的余弦相似度
支持批处理，所有参数前置，便于调参
"""

import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path
import torch.nn.functional as F

# ==================== 参数配置区 ====================
# 路径配置
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"          # 测试集所在目录
OUTPUT_DIR = PROJECT_ROOT / "results" / "o2"            # 模型和结果输出目录

# 模型与文件名称
MODEL_SUBDIR = "simcse_model"                            # 训练好的模型子目录（应与训练脚本 OUTPUT_SUBDIR 一致）
CENTER_VECTOR_FILE = "center_vector.npy"                 # 中心向量文件名

# 推理参数
BATCH_SIZE = 32                                           # 批处理大小（根据显存调整）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择设备
MAP_TO_100 = True                                         # 是否将相似度 [-1,1] 映射到 [0,100]
# ====================================================

def predict():
    print(f"使用设备: {DEVICE}")
    
    # 加载模型
    model_path = OUTPUT_DIR / MODEL_SUBDIR
    print(f"加载模型: {model_path}")
    model = SentenceTransformer(str(model_path))
    model.to(DEVICE)
    
    # 加载中心向量
    center_path = OUTPUT_DIR / CENTER_VECTOR_FILE
    print(f"加载中心向量: {center_path}")
    center = np.load(center_path)
    center_tensor = torch.tensor(center).to(DEVICE)
    
    # 加载测试集
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    texts = test_df['text'].tolist()
    print(f"测试集样本数: {len(texts)}")
    
    # 批量编码
    print("开始推理...")
    embeddings = model.encode(
        texts, 
        convert_to_tensor=True, 
        device=DEVICE,
        batch_size=BATCH_SIZE,
        show_progress_bar=True
    )
    
    # 计算余弦相似度
    # embeddings shape: (n, dim), center_tensor shape: (dim,)
    # 相似度 = 点积 / (模长乘积) = cos(emb, center)
    similarities = F.cosine_similarity(embeddings, center_tensor.unsqueeze(0), dim=1)
    sim_scores = similarities.cpu().numpy()
    
    # 可选映射到 0-100
    if MAP_TO_100:
        scores = (sim_scores + 1) / 2 * 100   # [-1,1] -> [0,100]
    else:
        scores = sim_scores
    
    # 保存结果
    result_df = pd.DataFrame({
        'text': texts,
        'score': scores
    })
    out_path = OUTPUT_DIR / "predictions.csv"
    result_df.to_csv(out_path, index=False)
    print(f"结果已保存至: {out_path}")
    print(f"分数统计: min={scores.min():.2f}, max={scores.max():.2f}, mean={scores.mean():.2f}")

if __name__ == "__main__":
    predict()