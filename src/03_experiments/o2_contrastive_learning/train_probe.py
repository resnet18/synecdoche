import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
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
from datasets import Dataset as HFDataset  # 新增导入
from sentence_transformers.models import Dense  # 投影头模块

# ==================== 参数配置区 ====================
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
BATCH_SIZE = 32
EPOCHS = 10
WARMUP_STEPS = 100
LEARNING_RATE = 1e-4
GENERATE_CENTER = True

# 早停参数
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 0.001
# ====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "final"
OUTPUT_DIR = PROJECT_ROOT / "results" / "o2_probe"  # 原来是 o2
OUTPUT_SUBDIR = "simcse_probe"  # 子目录改名，避免覆盖
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 配置日志
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.INFO)

# -------------------- 数据加载 --------------------
def load_data():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "validation.csv")
    return train_df['text'].tolist(), val_df['text'].tolist()

# -------------------- 转换为 Hugging Face Dataset --------------------
def create_dataset(sentences):
    """从句子列表创建包含 anchor 和 positive 列的 HF Dataset"""
    data = [{'anchor': s, 'positive': s} for s in sentences]  # SimCSE 无监督使用相同句子对
    return HFDataset.from_list(data)

# -------------------- 中心向量生成 --------------------
def generate_center_vector(model, sentences, device, output_path):
    print("生成中心向量（Probe 128维）...")
    embeddings = model.encode(sentences, convert_to_tensor=True, device=device)
    print(f"  向量维度验证：{embeddings.shape[1]}")  # 应该输出 128
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
    train_dataset = create_dataset(train_sentences)
    eval_dataset = create_dataset(val_sentences)

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(eval_dataset)}")

    # 加载模型
    model = SentenceTransformer(MODEL_NAME)
    # 冻结 MiniLM 所有参数（约2200万参数不再更新）
    for param in model.parameters():
        param.requires_grad = False
    
    # 添加轻量投影头：384维 -> 128维
    projection_head = Dense(
    in_features=384,
    out_features=128,
    activation_function=torch.nn.ReLU()
    )
    model.add_module('projection', projection_head)  # 加入模型序列
    
    # 验证可训练参数量（应该显示约50,000）
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"参数统计：总计 {total:,} | 可训练 {trainable:,} ({trainable/total*100:.2f}%)")
    model.to(device)

    # 定义损失函数
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # 设置训练参数
    # 处理 logging_dir 弃用警告：设置环境变量
    os.environ['TENSORBOARD_LOGGING_DIR'] = str(OUTPUT_DIR / "logs")
    args = SentenceTransformerTrainingArguments(
        output_dir=str(OUTPUT_DIR / OUTPUT_SUBDIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=torch.cuda.is_available(),          # 自动混合精度
        eval_strategy="epoch",                   # 新版参数名
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=50,
        report_to="none",
        # logging_dir 已移除，用环境变量代替
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
        loss=train_loss,
        callbacks=[early_stopping_callback],
    )

    # 开始训练
    print("开始训练 O2 (SimCSE)...")
    trainer.train()

    # 保存最终模型
    final_model_path = OUTPUT_DIR / OUTPUT_SUBDIR / "final_model"
    trainer.save_model(str(final_model_path))
    print(f"模型已保存至: {final_model_path}")

    # 生成中心向量
    if GENERATE_CENTER:
        generate_center_vector(model, train_sentences, device, OUTPUT_DIR)

if __name__ == "__main__":
    train_simcse()