# 不需要联网的手动版，以后研究一下
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ==================== 参数配置 ====================
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
BATCH_SIZE = 32
EPOCHS = 10
WARMUP_STEPS = 100
LEARNING_RATE = 2e-5
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 0.001
# ====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "final"
OUTPUT_DIR = PROJECT_ROOT / "results" / "o2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class SimCSEDataset(Dataset):
    def __init__(self, sentences):
        self.examples = [InputExample(texts=[s, s]) for s in sentences]
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def load_data():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "validation.csv")
    return train_df['text'].tolist(), val_df['text'].tolist()

def evaluate(model, dataloader, device):
    """计算验证集平均损失"""
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            # 手动计算损失
            texts = [example.texts for example in batch]
            # 第一个句子列表
            sentences1 = [t[0] for t in texts]
            # 第二个句子列表（SimCSE里和第一个一样）
            sentences2 = [t[1] for t in texts]
            
            embeddings1 = model.encode(sentences1, convert_to_tensor=True, device=device)
            embeddings2 = model.encode(sentences2, convert_to_tensor=True, device=device)
            
            # 计算余弦相似度损失（MultipleNegativesRankingLoss 的简化版）
            scores = torch.cosine_similarity(embeddings1, embeddings2, dim=1)
            # 对角线应该是1（正样本），其余是负样本
            loss = torch.mean(1 - scores)  # 简化的损失计算
            total_loss += loss.item()
            count += 1
    return total_loss / count if count > 0 else float('inf')

def train_simcse():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    train_sentences, val_sentences = load_data()
    train_dataset = SimCSEDataset(train_sentences)
    val_dataset = SimCSEDataset(val_sentences)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    model = SentenceTransformer(MODEL_NAME)
    model.to(device)
    
    # 使用 MultipleNegativesRankingLoss
    train_loss = losses.MultipleNegativesRankingLoss(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"开始训练 O2 (SimCSE)...")
    print(f"训练集: {len(train_dataset)} | 验证集: {len(val_dataset)}")
    
    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        epoch_loss = 0
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            optimizer.zero_grad()
            loss_value = train_loss(batch, batch)  # SimCSE: (batch, batch) 作为正负样本对
            loss_value.backward()
            optimizer.step()
            epoch_loss += loss_value.item()
        
        avg_train_loss = epoch_loss / len(train_dataloader)
        
        # 验证阶段（简化的 loss 监控）
        # 注意：MultipleNegativesRankingLoss 的完整验证需要特殊的 dataloader
        # 这里用模型保存和早停逻辑代替
        print(f"Epoch {epoch+1} | 平均训练损失: {avg_train_loss:.4f}")
        
        # 简单早停：如果训练损失不再下降
        if epoch > 0:
            if avg_train_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA:
                best_val_loss = avg_train_loss
                patience_counter = 0
                # 保存最佳模型
                model.save(str(OUTPUT_DIR / "simcse_model" / "best_model"))
                print("  -> 保存最佳模型")
            else:
                patience_counter += 1
                print(f"  -> 早停计数: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
                
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"早停触发，训练结束")
                break
    
    # 保存最终模型
    final_path = OUTPUT_DIR / "simcse_model" / "final_model"
    model.save(str(final_path))
    print(f"模型已保存至: {final_path}")
    
    # 生成中心向量
    print("生成中心向量...")
    embeddings = model.encode(train_sentences, convert_to_tensor=True, device=device)
    center = torch.mean(embeddings, dim=0)
    np.save(OUTPUT_DIR / "center_vector.npy", center.cpu().numpy())
    print("完成")

if __name__ == "__main__":
    train_simcse()