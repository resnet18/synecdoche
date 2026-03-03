"""
O2 推理: 计算与中心向量的余弦相似度
"""
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "o2"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

def predict():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载模型和中心向量
    model = SentenceTransformer(str(OUTPUT_DIR / "simcse_model"))
    model.to(device)
    center = np.load(OUTPUT_DIR / "center_vector.npy")
    center_tensor = torch.tensor(center).to(device)
    
    # 加载测试集
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    scores = []
    print("开始推理...")
    for text in test_df['text']:
        emb = model.encode(text, convert_to_tensor=True, device=device)
        # 余弦相似度
        sim = F.cosine_similarity(emb, center_tensor, dim=0)
        # 映射到 0-100 分
        score = (sim.item() + 1) / 2 * 100  # [-1,1] -> [0,100]
        scores.append(score)
    
    # 保存结果
    result_df = pd.DataFrame({
        'text': test_df['text'],
        'score': scores
    })
    result_df.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    print(f"结果已保存至: {OUTPUT_DIR / 'predictions.csv'}")

if __name__ == "__main__":
    predict()