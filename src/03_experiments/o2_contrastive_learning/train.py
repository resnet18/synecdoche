"""
O2: SimCSE 对比学习
使用 all-MiniLM-L6-v2 (22M) 作为基础模型
"""
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path

# 路径配置
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "results" / "o2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """加载 processed/train.csv 和 val.csv"""
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    return train_df, val_df

def train_simcse():
    # 加载基础模型 (22M参数，轻松装入16G显存)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # 检查GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    if device == "cuda":
        print(f"显卡: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    model.to(device)
    
    # 加载数据
    train_df, val_df = load_data()
    train_sentences = train_df['text'].tolist()
    
    # 构造 SimCSE 训练样本 (无监督: 同一文本不同dropout作为正样本对)
    # 实际上 sentence-transformers 的 MultipleNegativesRankingLoss 配合数据增强即可实现
    train_examples = [InputExample(texts=[s, s]) for s in train_sentences]
    train_dataloader = DataLoader(train_examples, batch_size=32, shuffle=True)
    
    # 损失函数: 多负例排序损失 (SimCSE核心)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    print("开始训练 O2 (SimCSE)...")
    print(f"训练集样本数: {len(train_examples)}")
    
    # 训练参数 (轻量级，快速验证)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,                    # 先跑3轮验证流程，正式实验可调到10-20
        warmup_steps=100,
        output_path=str(OUTPUT_DIR / "simcse_model"),
        show_progress_bar=True,
        save_best_model=True,
        # 验证集监控 (用验证集做Evaluator)
        # evaluator=evaluator,       # TODO: 后面加上早停逻辑
    )
    
    print(f"模型已保存至: {OUTPUT_DIR / 'simcse_model'}")
    
    # 保存中心向量 (用于后续推理)
    generate_center_vector(model, train_sentences, device)

def generate_center_vector(model, sentences, device):
    """计算训练集中心向量"""
    print("生成中心向量...")
    embeddings = model.encode(sentences, convert_to_tensor=True, device=device)
    center = torch.mean(embeddings, dim=0)
    np.save(OUTPUT_DIR / "center_vector.npy", center.cpu().numpy())
    print("中心向量已保存")

if __name__ == "__main__":
    train_simcse()