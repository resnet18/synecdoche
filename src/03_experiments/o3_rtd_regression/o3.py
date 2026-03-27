"""
O3: ELECTRA-small 零样本 (冻结Discriminator，无微调)
基于哈工大-讯飞联合实验室 chinese-electra-180g-small-discriminator (14M参数)
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"

from transformers import ElectraTokenizer, ElectraModel
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# === 模型参数（按需调整） ===
MODEL_NAME = 'hfl/chinese-electra-180g-small-discriminator'  # 14M参数，中文ELECTRA-small
BATCH_SIZE = 32  # 编码时的batch size
DEVICE = 'auto'  # 'auto', 'cuda', 或 'cpu'
MAX_LENGTH = 128  # ELECTRA-small最大序列长度（80字短文本绰绰有余）

# === 路径配置（自动计算） ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'o3')
os.makedirs(RESULTS_DIR, exist_ok=True)

# 合并训练集和验证集作为"阅读语料"（625句纯文本）
TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'final', 'train.csv')
VALIDATION_PATH = os.path.join(BASE_DIR, 'data', 'final', 'validation.csv')
TEST_PATH = os.path.join(BASE_DIR, 'data', 'test', 'test_positive.csv')

OUTPUT_PRED = os.path.join(RESULTS_DIR, 'predictions.csv')

print("=== O3 ELECTRA-small 零样本基线（冻结参数，读全量建中心） ===")

# 1. 设备检查
if DEVICE == 'auto':
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = DEVICE

print(f"使用设备: {device}")
if device == "cuda":
    print(f"显卡: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 2. 加载模型（完全冻结！纯零样本）
print(f"加载模型: {MODEL_NAME}...")
tokenizer = ElectraTokenizer.from_pretrained(MODEL_NAME)
model = ElectraModel.from_pretrained(MODEL_NAME).to(device)

# 硬冻结：推理模式+禁止梯度
model.eval()
for param in model.parameters():
    param.requires_grad = False

print(f"模型已冻结 | 参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# 3. 定义编码函数（替代SentenceTransformer的encode）
@torch.no_grad()
def encode_texts(texts, convert_to_tensor=True):
    """
    批量编码文本，返回Mean Pooling 向量 (256维)
    """
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True
        ).to(device)
        
        outputs = model(**inputs)
        # 替换原来的 cls_emb 提取
        mean_emb = outputs.last_hidden_state.mean(dim=1)  # (batch, 256)
        
        if convert_to_tensor:
            all_embeddings.append(mean_emb)
        else:
            all_embeddings.append(mean_emb.cpu().numpy())
    
    if convert_to_tensor:
        return torch.cat(all_embeddings, dim=0)
    else:
        return np.vstack(all_embeddings)

# 4. 读取数据（合并train + validation，共625句）
print("读取训练语料（train + validation）...")
train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VALIDATION_PATH)
test_df = pd.read_csv(TEST_PATH)

# 合并全量阅读语料
all_train_texts = pd.concat([train_df, val_df])['text'].astype(str).tolist()
test_texts = test_df['text'].astype(str).tolist()

print(f"阅读语料（train+validation）: {len(all_train_texts)} 句")
print(f"测试集: {len(test_texts)} 句")

# 5. 计算中心向量（全量语料的平均embedding）
print("编码阅读语料并计算中心向量...")
train_embs = encode_texts(all_train_texts, convert_to_tensor=True)
center = train_embs.mean(dim=0)
print(f"中心向量计算完成 (dim={center.shape[0]})")

center_path = os.path.join(RESULTS_DIR, 'center_vector.npy')
np.save(center_path, center.cpu().numpy())
print(f"中心向量已保存至: {center_path}")

# 6. 测试集推理（batch处理，与B5逻辑一致）
print("预测测试集...")
scores = []

# batch处理测试集以提高效率
for idx in range(0, len(test_texts), BATCH_SIZE):
    batch_texts = test_texts[idx:idx+BATCH_SIZE]
    batch_embs = encode_texts(batch_texts, convert_to_tensor=True)
    
    # 计算每个样本与中心的余弦相似度
    for emb in batch_embs:
        sim = F.cosine_similarity(emb, center, dim=0)
        # [-1,1] -> [0,100]，与B5保持一致
        score = (sim.item() + 1) / 2 * 100
        scores.append(score)
    
    # 进度打印（每32条或最后一条）
    current = min(idx + BATCH_SIZE, len(test_texts))
    if current % 32 == 0 or current == len(test_texts):
        print(f"  已处理 {current}/{len(test_texts)} 条")

# 7. 保存结果（与B5完全兼容的格式）
results = pd.DataFrame({
    'text': test_texts,
    'o3_score': scores,  # 对应B5的b5_score
    'similarity_raw': [(s/100)*2 - 1 for s in scores]  # 保留原始相似度[-1,1]
})
results.to_csv(OUTPUT_PRED, index=False, encoding='utf-8-sig')

# 8. 统计信息
print(f"\n结果保存至: {RESULTS_DIR}")
print(f"分数统计: min={min(scores):.2f}, max={max(scores):.2f}, mean={sum(scores)/len(scores):.2f}")
print("=== O3 完成 ===")