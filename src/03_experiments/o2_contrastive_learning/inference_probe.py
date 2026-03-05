"""
O2-Probe 推理: 冻结 MiniLM + 投影头 (128维)
"""
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path
import torch.nn.functional as F

# ==================== 参数配置区 ====================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "test"
OUTPUT_DIR = PROJECT_ROOT / "results" / "o2_probe"  # 关键修改：o2_probe

# 与 train_probe.py 的 OUTPUT_SUBDIR 一致
MODEL_SUBDIR = "simcse_probe/final_model"  # 关键修改：simcse_probe
CENTER_VECTOR_FILE = "center_vector.npy"

BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAP_TO_100 = True
# ====================================================

def predict():
    print(f"使用设备: {DEVICE}")
    
    # 加载模型（应包含 projection 层）
    model_path = OUTPUT_DIR / MODEL_SUBDIR
    print(f"加载 Probe 模型: {model_path}")
    model = SentenceTransformer(str(model_path))
    model.to(DEVICE)
    
    # 加载中心向量（128维）
    center_path = OUTPUT_DIR / CENTER_VECTOR_FILE
    print(f"加载中心向量: {center_path}")
    center = np.load(center_path)
    center_tensor = torch.tensor(center).to(DEVICE)
    print(f"  中心向量维度: {center.shape[0]}")  # 应显示 128
    
    # 加载测试集
    test_df = pd.read_csv(DATA_DIR / "test_positive.csv")
    texts = test_df['text'].tolist()
    print(f"测试集样本数: {len(texts)}")
    
    # 编码（自动经过 projection 头，输出 128 维）
    print("开始推理...")
    embeddings = model.encode(
        texts, 
        convert_to_tensor=True, 
        device=DEVICE,
        batch_size=BATCH_SIZE,
        show_progress_bar=True
    )
    
    # 关键验证：维度必须匹配
    if embeddings.shape[1] != center.shape[0]:
        raise ValueError(f"维度不匹配！Embeddings: {embeddings.shape[1]}, Center: {center.shape[0]}")
    
    # 计算余弦相似度
    similarities = F.cosine_similarity(embeddings, center_tensor.unsqueeze(0), dim=1)
    sim_scores = similarities.cpu().numpy()
    
    if MAP_TO_100:
        scores = (sim_scores + 1) / 2 * 100
    else:
        scores = sim_scores
    
    # 保存
    result_df = pd.DataFrame({
        'text': texts,
        'o2_score': scores  # 与 b4a.py 等保持一致，用方法名作为列名
    })
    out_path = OUTPUT_DIR / "predictions.csv"
    result_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"结果已保存: {out_path}")
    print(f"分数统计: min={scores.min():.2f}, max={scores.max():.2f}, mean={scores.mean():.2f}")

if __name__ == "__main__":
    predict()