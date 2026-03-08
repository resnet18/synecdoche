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
from datasets import Dataset as HFDataset

# ==================== 参数配置区 ====================
MODEL_NAME = 'hfl/chinese-roberta-wwm-ext-large'  # 中文纯MLM基座，355M参数
# 备选（显存不足时）: 'hfl/chinese-roberta-wwm-ext' (102M参数，base版)

BATCH_SIZE = 4                                               # 355M模型，16G显存安全值
MAX_SEQ_LENGTH = 128                                         # 显存保护
EPOCHS = 10                                                  # 最大轮数（早停可能提前结束）
WARMUP_STEPS = 0                                             # 预热步数
LEARNING_RATE = 1e-5                                         # 默认SimCSE推荐学习率
OUTPUT_SUBDIR = "roberta_simcse"                             # 保存子目录
GENERATE_CENTER = True                                       # 生成中心向量
WEIGHT_DECAY = 0.1

# 早停参数
EARLY_STOPPING_PATIENCE = 5                                  # 连续几轮无改善则停
EARLY_STOPPING_MIN_DELTA = 0.001                             # 最小改善阈值
# ====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "final"                   # 注意：使用final目录（脱敏数据）
OUTPUT_DIR = PROJECT_ROOT / "results" / "b6"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 配置日志（同时输出到控制台和文件）
log_file = OUTPUT_DIR / "training.log"
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),  # 原有：控制台输出
        logging.FileHandler(log_file, encoding='utf-8')  # 新增：保存到文件
    ]
)
# -------------------- 数据加载 --------------------
def load_data():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "validation.csv")
    return train_df['text'].tolist(), val_df['text'].tolist()

# -------------------- 转换为 Hugging Face Dataset --------------------
def create_dataset(sentences):
    """SimCSE无监督：anchor和positive是同一个句子（经过不同dropout）"""
    data = [{'anchor': s, 'positive': s} for s in sentences]
    return HFDataset.from_list(data)

# -------------------- 中心向量生成 --------------------
def generate_center_vector(model, sentences, device, output_path):
    print("生成中心向量...")
    embeddings = model.encode(
        sentences, 
        convert_to_tensor=True, 
        device=device,
        show_progress_bar=True,
        batch_size=16  # 推理可以用较大batch
    )
    center = torch.mean(embeddings, dim=0)
    np.save(output_path / "center_vector.npy", center.cpu().numpy())
    print(f"中心向量已保存: {output_path / 'center_vector.npy'}")

# -------------------- 主训练函数 --------------------
def train_b6():
    # 设备检测
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    if device == "cuda":
        print(f"显卡: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"显存总量: {total_mem:.1f} GB")
        torch.cuda.empty_cache()

    # 加载数据
    train_sentences, val_sentences = load_data()
    train_dataset = create_dataset(train_sentences)
    eval_dataset = create_dataset(val_sentences)

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(eval_dataset)}")

    # 加载模型（中文RoBERTa-large，纯MLM基座）
    print(f"\n加载模型: {MODEL_NAME}")
    print("注意：使用纯MLM预训练基座（355M参数）+ Mean Pooling...")
    
    try:
        model = SentenceTransformer(MODEL_NAME)
        model.max_seq_length = MAX_SEQ_LENGTH  # 限制序列长度
        model.to(device)
    except Exception as e:
        print(f"模型加载失败: {e}")
        raise

    # 显存状态检查
    if device == "cuda":
        allocated = torch.cuda.memory_allocated() / 1024**2
        print(f"模型加载后显存: {allocated:.0f} MB")

    # 定义SimCSE损失函数（MultipleNegativesRankingLoss）
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # 设置训练参数
    os.environ['TENSORBOARD_LOGGING_DIR'] = str(OUTPUT_DIR / "logs")
    
    args = SentenceTransformerTrainingArguments(
        output_dir=str(OUTPUT_DIR / OUTPUT_SUBDIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fp16=torch.cuda.is_available(),          # 自动混合精度（节省显存）
        eval_strategy="epoch",                    # 每个epoch评估验证集
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        logging_strategy="epoch",
        logging_steps=50,
        report_to="none",
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

    # 开始训练（带OOM捕获）
    print(f"\n开始训练 B6 (Chinese-RoBERTa-large SimCSE)...")
    print(f"模型参数: ~355M")
    print(f"Batch size: {BATCH_SIZE}, Max seq length: {MAX_SEQ_LENGTH}")
    print(f"学习率: {LEARNING_RATE}, 早停耐心: {EARLY_STOPPING_PATIENCE}")
    
    try:
        trainer.train()
        
        # 保存最终模型
        final_model_path = OUTPUT_DIR / OUTPUT_SUBDIR / "final_model"
        trainer.save_model(str(final_model_path))
        print(f"\n✓ 模型已保存至: {final_model_path}")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            print("\n" + "="*50)
            print("!!! 显存不足 (Out of Memory) !!!")
            print("="*50)
            print("建议降级方案：")
            print("1. 修改 BATCH_SIZE = 2 甚至 1")
            print("2. 修改 MAX_SEQ_LENGTH = 64")
            print("3. 换用轻量模型: 'hfl/chinese-roberta-wwm-ext' (102M参数)")
            print("4. 如果还不行，跳过B6-simcse，只跑B6-frozen")
            print("="*50)
            raise e
        else:
            raise e

    # 生成中心向量
    if GENERATE_CENTER:
        print("\n生成训练集中心向量...")
        generate_center_vector(model, train_sentences, device, OUTPUT_DIR)

if __name__ == "__main__":
    train_b6()