"""
B6: RoBERTa-large + SimCSE (340M参数)
与 O2 同范式但更大，对比验证规模效应
"""
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "results" / "b6"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def train_b6():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载 RoBERTa-large (340M参数)
    print("加载 RoBERTa-large (340M)...")
    model = SentenceTransformer('sentence-transformers/nli-roberta-large-v2')  # 或 simcse-roberta-large
    model.to(device)
    
    # 显存检查 (340M 应该能装进16G，但batch_size要调小)
    if device == "cuda":
        print(f"显存已分配: {torch.cuda.memory_allocated() / 1024**2:.0f} MB")
    
    # 数据加载 (同 O2)
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    train_examples = [InputExample(texts=[s, s]) for s in train_df['text']]
    
    # batch_size 调小至 8 或 16，避免340M+16G显存 OOM
    train_dataloader = DataLoader(train_examples, batch_size=8, shuffle=True)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    print("开始训练 B6...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        warmup_steps=50,
        output_path=str(OUTPUT_DIR / "roberta_simcse"),
        show_progress_bar=True
    )
    print("B6 训练完成")

if __name__ == "__main__":
    train_b6()