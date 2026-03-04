"""
O2: SimCSE 对比学习 + 早停
使用 all-MiniLM-L6-v2 (22M) 作为基础模型
支持验证集早停和每个epoch损失打印
"""

import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers import evaluation  # 用于 EarlyStopping
from sentence_transformers.evaluation import SentenceEvaluator   # 新增导入
import logging  # 新增导入
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path

# ==================== 参数配置区 ====================
# 模型与训练基本参数
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'  # 基础模型名称
BATCH_SIZE = 32                                        # 批大小
EPOCHS = 10                                             # 最大训练轮数（早停可能提前结束）
WARMUP_STEPS = 100                                      # 预热步数
LEARNING_RATE = 2e-5                                    # 学习率
OUTPUT_SUBDIR = "simcse_model"                          # 模型保存子目录名
SAVE_BEST_MODEL = True                                  # 是否保存最佳模型
GENERATE_CENTER = True                                  # 是否生成中心向量

# 早停参数
USE_EARLY_STOPPING = True                               # 是否启用早停
EARLY_STOPPING_PATIENCE = 3                              # 验证损失连续多少轮不下降就停止
EARLY_STOPPING_MIN_DELTA = 0.001                         # 最小改善阈值（损失下降小于此值视为无改善）
EVALUATE_EVERY_EPOCH = True                              # 每个epoch评估验证损失（用于早停监控）
# ====================================================

# 路径配置
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # 第3个父目录（从0开始计数）
DATA_DIR = PROJECT_ROOT / "data" / "final"
OUTPUT_DIR = PROJECT_ROOT / "results" / "o2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 自定义 LossEvaluator 类（新增） ====================
class LossEvaluator(SentenceEvaluator):
    """自定义评估器，用于在验证集上计算 MultipleNegativesRankingLoss"""
    def __init__(self, dataloader, loss_model, name=""):
        self.dataloader = dataloader
        self.loss_model = loss_model
        self.name = name
        self.best_loss = float('inf')

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.dataloader:
                # 将 InputExample 列表转换为模型输入格式
                features, labels = model.smart_batching_collate(batch)
                # 将特征移到模型所在设备
                features = [f.to(model.device) for f in features]
                # 计算损失（labels 也会自动处理）
                loss_value = self.loss_model(features, labels=labels)
                total_loss += loss_value.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        logging.info(f"Evaluation on {self.name} dataset (epoch {epoch}, steps {steps}): loss = {avg_loss:.6f}")
        return avg_loss
# =========================================================================

def load_data():
    """加载训练集和验证集"""
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "validation.csv")
    return train_df, val_df

def train_simcse():
    # 加载基础模型
    model = SentenceTransformer(MODEL_NAME)
    
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
    val_sentences = val_df['text'].tolist()
    
    # 构造训练集样本
    train_examples = [InputExample(texts=[s, s]) for s in train_sentences]
    train_dataloader = DataLoader(train_examples, batch_size=BATCH_SIZE, shuffle=True)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # 构造验证集样本（同样使用无监督方式）
    val_examples = [InputExample(texts=[s, s]) for s in val_sentences]
    val_dataloader = DataLoader(val_examples, batch_size=BATCH_SIZE, shuffle=False)
    
    # ----- 设置 evaluator 和早停 -----
    evaluator = None
    early_stopping = None
    if EVALUATE_EVERY_EPOCH:
        # 验证集上的损失评估器
         # 使用自定义的 LossEvaluator，而不是从库导入
        evaluator = LossEvaluator(val_dataloader, loss_model=train_loss, name='val_loss')
        # 每个epoch的步数（用于设置 evaluation_steps）
        steps_per_epoch = len(train_dataloader)
        
        if USE_EARLY_STOPPING:
            # 检查 EarlyStopping 是否存在
            if hasattr(evaluation, 'EarlyStopping'):
                early_stopping = evaluation.EarlyStopping(
                    evaluator=evaluator,
                    patience=EARLY_STOPPING_PATIENCE,
                    min_delta=EARLY_STOPPING_MIN_DELTA
                )
                print(f"早停已启用：耐心={EARLY_STOPPING_PATIENCE}，最小改善={EARLY_STOPPING_MIN_DELTA}")
            else:
                print("警告：当前版本不支持 EarlyStopping，早停功能已禁用。")
                USE_EARLY_STOPPING = False
    else:
        steps_per_epoch = 0
    
    print("开始训练 O2 (SimCSE)...")
    print(f"训练集样本数: {len(train_examples)}")
    print(f"验证集样本数: {len(val_examples)}")
    if EVALUATE_EVERY_EPOCH:
        print(f"每个epoch评估一次验证损失（steps_per_epoch={steps_per_epoch}）")
    
    # 训练（包含早停）
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=WARMUP_STEPS,
        optimizer_params={'lr': LEARNING_RATE},
        output_path=str(OUTPUT_DIR / OUTPUT_SUBDIR),
        show_progress_bar=True,
        save_best_model=SAVE_BEST_MODEL,
        evaluator=evaluator,                 # 验证集评估器
        evaluation_steps=steps_per_epoch,    # 每个epoch评估一次
        early_stopping=early_stopping        # 早停对象
    )
    
    print(f"模型已保存至: {OUTPUT_DIR / OUTPUT_SUBDIR}")
    
    # 生成中心向量（用于后续推理）
    if GENERATE_CENTER:
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