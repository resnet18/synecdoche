import torch
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments
)
from transformers.trainer_callback import EarlyStoppingCallback
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# ==================== 参数配置区 ====================
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
BATCH_SIZE = 32
EPOCHS = 10
WARMUP_STEPS = 100
LEARNING_RATE = 2e-5
OUTPUT_SUBDIR = "simcse_model"
GENERATE_CENTER = True

# 早停参数
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 0.001
# ====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "final"
OUTPUT_DIR = PROJECT_ROOT / "results" / "o2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 配置日志
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.INFO)

# -------------------- 自定义数据集 --------------------
class SimCSEDataset(Dataset):
    """将句子列表包装成 InputExample 的数据集"""
    def __init__(self, sentences):
        self.examples = [InputExample(texts=[s, s]) for s in sentences]  # SimCSE 无监督：句子对 (s, s)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# -------------------- 数据加载 --------------------
def load_data():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "validation.csv")
    return train_df['text'].tolist(), val_df['text'].tolist()

# -------------------- 中心向量生成 --------------------
def generate_center_vector(model, sentences, device, output_path):
    print("生成中心向量...")
    embeddings = model.encode(sentences, convert_to_tensor=True, device=device)
    center = torch.mean(embeddings, dim=0)
    np.save(output_path / "center_vector.npy", center.cpu().numpy())
    print("中心向量已保存")

# -------------------- 主训练函数 --------------------
def train_simcse():
    # 设备检测
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    if device == "cuda":
        print(f"显卡: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 加载数据
    train_sentences, val_sentences = load_data()
    train_dataset = SimCSEDataset(train_sentences)
    eval_dataset = SimCSEDataset(val_sentences)

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(eval_dataset)}")

    # 加载模型
    model = SentenceTransformer(MODEL_NAME)
    model.to(device)

    # 定义损失函数
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # 设置训练参数
    args = SentenceTransformerTrainingArguments(
        output_dir=str(OUTPUT_DIR / OUTPUT_SUBDIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=torch.cuda.is_available(),          # 自动混合精度（如果 GPU 支持）
        evaluation_strategy="epoch",              # 每个 epoch 结束后评估
        save_strategy="epoch",                    # 每个 epoch 保存一次
        load_best_model_at_end=True,               # 训练结束后加载最佳模型
        metric_for_best_model="eval_loss",         # 监控验证损失
        greater_is_better=False,                   # 损失越小越好
        save_total_limit=2,                         # 只保留最后两个 checkpoint
        logging_dir=str(OUTPUT_DIR / "logs"),      # 日志目录
        logging_steps=50,
        report_to="none",                            # 不向外部平台报告
    )

    # 创建早停回调
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_threshold=EARLY_STOPPING_MIN_DELTA
    )

    # 创建 Trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,                     # 传入损失函数
        callbacks=[early_stopping_callback],  # 早停回调
    )

    # 开始训练
    print("开始训练 O2 (SimCSE)...")
    trainer.train()

    # 训练完成后，最佳模型已自动加载，保存最终模型（可选）
    final_model_path = OUTPUT_DIR / OUTPUT_SUBDIR / "final_model"
    trainer.save_model(str(final_model_path))
    print(f"模型已保存至: {final_model_path}")

    # 生成中心向量（使用训练好的模型）
    if GENERATE_CENTER:
        generate_center_vector(model, train_sentences, device, OUTPUT_DIR)

if __name__ == "__main__":
    train_simcse()