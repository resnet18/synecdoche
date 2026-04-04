"""
B5: MiniLM 零样本 (直接取预训练embedding，无微调)
- 最终测试集推理版本：直接加载预计算锚点，零训练
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 第一行！
os.environ['HF_HUB_OFFLINE'] = '0'  # 允许联网，但走镜像
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch

# ---------- 显存限制代码 ----------
if os.getenv('CUDA_MEMORY_FRACTION'):
    torch.cuda.set_per_process_memory_fraction(float(os.getenv('CUDA_MEMORY_FRACTION')))
    print(f"显存限制: {float(os.getenv('CUDA_MEMORY_FRACTION'))*100:.0f}%")
# -------------------------------------

# === 模型参数 ===
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
BATCH_SIZE = 32
DEVICE = 'auto'

# === 路径配置（4层dirname适配final_validation路径） ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 输入：预实验计算好的中心向量（只读）
PRETRAIN_DIR = os.path.join(BASE_DIR, 'results', 'b5')
CENTER_PATH = os.path.join(PRETRAIN_DIR, 'center_vector.npy')

# 输入：最终测试集（210句完整测试集）
TEST_PATH = os.path.join(BASE_DIR, 'data', 'test', 'test_full.csv')

# 输出：最终验证结果目录
OUTPUT_DIR = os.path.join(BASE_DIR, 'results', 'final_validation', 'b5')
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PRED = os.path.join(OUTPUT_DIR, 'predictions.csv')

print("=== B5 MiniLM零样本基线 - 最终测试集推理 ===")

# 1. 设备检查
if DEVICE == 'auto':
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = DEVICE

print(f"使用设备: {device}")
if device == "cuda":
    print(f"显卡: {torch.cuda.get_device_name(0)}")

# 2. 加载预训练模型（无微调，零样本）
print(f"加载模型: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
model.to(device)

# 3. 加载预计算的风格锚点（中心向量）
print(f"加载预计算中心向量: {CENTER_PATH}")
center = np.load(CENTER_PATH)
center_tensor = torch.from_numpy(center).to(device)
print(f"锚点向量维度: {center.shape[0]}")

# 4. 读取最终测试集（210句，无需读取train/val）
test_df = pd.read_csv(TEST_PATH)
test_texts = test_df['text'].astype(str).tolist()
print(f"测试集规模: {len(test_texts)}句")

# 5. 测试集推理（batch编码）
print("编码测试集并计算偏离度...")
test_embs = model.encode(
    test_texts,
    convert_to_tensor=True,
    device=device,
    batch_size=BATCH_SIZE,
    show_progress_bar=True
)

# 计算余弦相似度（与锚点的偏离度）
sims = F.cosine_similarity(test_embs, center_tensor.unsqueeze(0), dim=1)
similarities = sims.cpu().numpy()

# 映射到0-10分（与项目其他基线保持一致）
# 原映射: (sim + 1) / 2 * 100 -> 0-100
# 新映射: (sim + 1) / 2 * 10 -> 0-10
scores = (similarities + 1) / 2 * 10

# 6. 保存结果（等待人类基准对齐）
results = pd.DataFrame({
    'text_id': test_df.get('text_id', range(len(test_texts))),
    'text': test_texts,
    'b5_score': scores,
    'cosine_sim_raw': similarities  # 保留原始[-1,1]范围相似度
})
results.to_csv(OUTPUT_PRED, index=False, encoding='utf-8-sig')

print(f"\n结果已保存: {OUTPUT_PRED}")
print(f"分数统计: min={scores.min():.2f}, max={scores.max():.2f}, mean={scores.mean():.2f}, std={scores.std():.2f}")
print(f"待人类基准打分完成后，合并计算Spearman相关性")

# ========== 显存报告（加在这里）==========
import torch
from pathlib import Path
import os

if torch.cuda.is_available():
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    # 从环境变量获取路径（父脚本传入），或默认当前目录
    peak_file = Path(os.environ.get('PEAK_FILE_PATH', '_gpu_peak_temp.txt'))
    peak_file.write_text(str(peak_mb))
    print(f"[GPU] 峰值显存: {peak_mb:.0f} MB")
else:
    # 即使没有GPU，也写入0，避免父脚本读取失败
    peak_file = Path(os.environ.get('PEAK_FILE_PATH', '_gpu_peak_temp.txt'))
    peak_file.write_text("0")