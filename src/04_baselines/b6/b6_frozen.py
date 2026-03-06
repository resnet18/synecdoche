"""
B6_frozen: RoBERTa-large 零样本 (直接取预训练embedding，无微调)
与 B5 同范式但更大，验证规模效应
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # ← 第一行！
os.environ['HF_HUB_OFFLINE'] = '0'  # 强制使用本地缓存，不联网
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch
from pathlib import Path

# === 模型参数 ===
MODEL_NAME = 'hfl/chinese-roberta-wwm-ext-large'  # 中文纯MLM基座，355M参数
BATCH_SIZE = 16  # RoBERTa-large更大，适当减小batch size
DEVICE = 'auto'  # 'auto', 'cuda', 或 'cpu'

# === 路径配置 ===
BASE_DIR = r'D:\Projects\synecdoche'
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'b6_frozen')  # 注意：b6_frozen子目录，避免和b6混淆
os.makedirs(RESULTS_DIR, exist_ok=True)

# 数据路径（同B5）
TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'final', 'train.csv')
VALIDATION_PATH = os.path.join(BASE_DIR, 'data', 'final', 'validation.csv')
TEST_PATH = os.path.join(BASE_DIR, 'data', 'test', 'test_positive.csv')

OUTPUT_PRED = os.path.join(RESULTS_DIR, 'predictions.csv')

print("=== B6 RoBERTa-large 零样本基线（无微调，读全量建中心） ===")

# 1. 设备检查
if DEVICE == 'auto':
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = DEVICE

print(f"使用设备: {device}")
if device == "cuda":
    print(f"显卡: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 2. 加载模型（不训练！纯零样本）
print(f"加载模型: {MODEL_NAME}...")
print("注意：这是纯MLM预训练基座，无句子级微调...")
model = SentenceTransformer(MODEL_NAME)
model.to(device)

# 3. 读取数据（合并train + validation）
print("读取训练语料（train + validation）...")
train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VALIDATION_PATH)
test_df = pd.read_csv(TEST_PATH)

all_train_texts = pd.concat([train_df, val_df])['text'].astype(str).tolist()
test_texts = test_df['text'].astype(str).tolist()

print(f"阅读语料（train+validation）: {len(all_train_texts)} 句")
print(f"测试集: {len(test_texts)} 句")

# 4. 计算中心向量
print("编码阅读语料并计算中心向量...")
train_embs = model.encode(
    all_train_texts, 
    convert_to_tensor=True, 
    device=device,
    batch_size=BATCH_SIZE,
    show_progress_bar=True
)
center = train_embs.mean(dim=0)
print(f"中心向量计算完成，维度: {center.shape[0]}")

center_path = os.path.join(RESULTS_DIR, 'center_vector.npy')
np.save(center_path, center.cpu().numpy())
print(f"中心向量已保存至: {center_path}")

# 5. 测试集推理
print("预测测试集...")
scores = []
for idx, text in enumerate(test_texts):
    emb = model.encode(text, convert_to_tensor=True, device=device)
    sim = F.cosine_similarity(emb, center, dim=0)
    score = (sim.item() + 1) / 2 * 100  # [-1,1] -> [0,100]
    scores.append(score)
    
    if (idx + 1) % 10 == 0:
        print(f"  已处理 {idx + 1}/{len(test_texts)} 条")

# 6. 保存结果
results = pd.DataFrame({
    'text': test_texts,
    'b6_frozen_score': scores,  # 列名区分
    'similarity_raw': [(s/100)*2 - 1 for s in scores]
})
results.to_csv(OUTPUT_PRED, index=False, encoding='utf-8-sig')

# 7. 统计信息
print(f"\n结果保存至: {RESULTS_DIR}")
print(f"分数统计: min={min(scores):.2f}, max={max(scores):.2f}, mean={sum(scores)/len(scores):.2f}")
print("=== 完成 ===")