"""
O3: ELECTRA-small 零样本 (冻结Discriminator) - 最终测试集推理
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

# ---------- 显存限制代码 ----------
if os.getenv('CUDA_MEMORY_FRACTION'):
    torch.cuda.set_per_process_memory_fraction(float(os.getenv('CUDA_MEMORY_FRACTION')))
    print(f"显存限制: {float(os.getenv('CUDA_MEMORY_FRACTION'))*100:.0f}%")
# -------------------------------------

# === 路径配置（4层dirname适配final_validation路径） ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 输入：预实验计算好的中心向量（只读）
PRETRAIN_DIR = os.path.join(BASE_DIR, 'results', 'o3')
CENTER_PATH = os.path.join(PRETRAIN_DIR, 'center_vector.npy')

# 输入：最终测试集（210句完整测试集）
TEST_PATH = os.path.join(BASE_DIR, 'data', 'test', 'test_full.csv')

# 输出：最终验证结果目录
OUTPUT_DIR = os.path.join(BASE_DIR, 'results', 'final_validation', 'o3')
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PRED = os.path.join(OUTPUT_DIR, 'predictions.csv')

# === 模型参数 ===
MODEL_NAME = 'hfl/chinese-electra-180g-small-discriminator'  # 14M参数
BATCH_SIZE = 32
DEVICE = 'auto' 
MAX_LENGTH = 128

print("=== O3 ELECTRA-small 零样本基线 - 最终测试集推理 ===")

# 1. 设备检查
if DEVICE == 'auto':
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = DEVICE

print(f"使用设备: {device}")
if device == "cuda":
    print(f"显卡: {torch.cuda.get_device_name(0)}")

# 2. 加载模型（冻结参数，零样本）
print(f"加载模型: {MODEL_NAME}")
tokenizer = ElectraTokenizer.from_pretrained(MODEL_NAME)
model = ElectraModel.from_pretrained(MODEL_NAME).to(device)

model.eval()
for param in model.parameters():
    param.requires_grad = False

print(f"模型已冻结 | 参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# 3. 定义编码函数（Mean Pooling）
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
        mean_emb = outputs.last_hidden_state.mean(dim=1)  # (batch, 256)
        
        if convert_to_tensor:
            all_embeddings.append(mean_emb)
        else:
            all_embeddings.append(mean_emb.cpu().numpy())
    
    if convert_to_tensor:
        return torch.cat(all_embeddings, dim=0)
    else:
        return np.vstack(all_embeddings)

# 4. 加载预计算的风格锚点（中心向量）
print(f"加载预计算中心向量: {CENTER_PATH}")
center = np.load(CENTER_PATH)
center_tensor = torch.from_numpy(center).to(device)
print(f"锚点向量维度: {center.shape[0]} (ELECTRA-small hidden size)")

# 5. 读取最终测试集（210句）
test_df = pd.read_csv(TEST_PATH)
test_texts = test_df['text'].astype(str).tolist()
print(f"测试集规模: {len(test_texts)}句")

# 6. 测试集推理（批处理）
print("编码测试集并计算偏离度...")
test_embs = encode_texts(test_texts, convert_to_tensor=True)

# 计算余弦相似度（与锚点的偏离度）
similarities = F.cosine_similarity(test_embs, center_tensor.unsqueeze(0), dim=1)
sim_scores = similarities.cpu().numpy()

# 映射到0-10分（与项目其他基线保持一致）
# 原预实验: (sim + 1) / 2 * 100 -> 0-100
# 现统一为: (sim + 1) / 2 * 10 -> 0-10
scores = (sim_scores + 1) / 2 * 10

# 7. 保存结果（等待人类基准对齐）
results = pd.DataFrame({
    'text_id': test_df.get('text_id', range(len(test_texts))),
    'text': test_texts,
    'o3_score': scores,
    'cosine_sim_raw': sim_scores  # 保留原始[-1,1]范围相似度
})
results.to_csv(OUTPUT_PRED, index=False, encoding='utf-8-sig')

print(f"\n结果已保存: {OUTPUT_PRED}")
print(f"分数统计: min={scores.min():.2f}, max={scores.max():.2f}, mean={scores.mean():.2f}, std={scores.std():.2f}")
print(f"待人类基准打分完成后，合并计算Spearman相关性")

# ========== 显存报告 ==========
import torch
from pathlib import Path
import os

if torch.cuda.is_available():
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    peak_file = Path(os.environ.get('PEAK_FILE_PATH', '_gpu_peak_temp.txt'))
    peak_file.write_text(str(peak_mb))
    print(f"[GPU] 峰值显存: {peak_mb:.0f} MB")
else:
    peak_file = Path(os.environ.get('PEAK_FILE_PATH', '_gpu_peak_temp.txt'))
    peak_file.write_text("0")