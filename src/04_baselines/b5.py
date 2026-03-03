"""
B5: MiniLM 零样本 (直接取预训练embedding，无微调)
"""
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "results" / "b5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def b5_baseline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"B5 零样本基线 - 设备: {device}")
    
    # 加载预训练模型 (不训练！)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    model.to(device)
    
    # 加载数据
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    # 计算训练集中心
    print("计算训练集中心向量...")
    train_embs = model.encode(train_df['text'].tolist(), convert_to_tensor=True, device=device)
    center = train_embs.mean(dim=0)
    
    # 推理测试集
    scores = []
    for text in test_df['text']:
        emb = model.encode(text, convert_to_tensor=True, device=device)
        sim = F.cosine_similarity(emb, center, dim=0)
        score = (sim.item() + 1) / 2 * 100
        scores.append(score)
    
    # 保存
    pd.DataFrame({
        'text': test_df['text'],
        'score': scores
    }).to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    
    print(f"B5 完成，结果保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    b5_baseline()