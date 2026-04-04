"""
B6: Chinese-RoBERTa-large SimCSE - 最终测试集推理
使用预实验训练好的模型，直接加载中心向量，零重新训练
"""

import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import os

# ---------- 显存限制代码 ----------
if os.getenv('CUDA_MEMORY_FRACTION'):
    torch.cuda.set_per_process_memory_fraction(float(os.getenv('CUDA_MEMORY_FRACTION')))
    print(f"显存限制: {float(os.getenv('CUDA_MEMORY_FRACTION'))*100:.0f}%")
# -------------------------------------

# ==================== 参数配置区 ====================
# 路径配置（4层dirname适配final_validation路径）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 输入：预实验训练好的模型和锚点（只读）
PRETRAIN_DIR = os.path.join(BASE_DIR, 'results', 'b6')
MODEL_SUBDIR = "roberta_simcse/final_model"  # Trainer保存的子目录
CENTER_VECTOR_FILE = "center_vector.npy"

# 输入：最终测试集（210句完整测试集）
TEST_PATH = os.path.join(BASE_DIR, 'data', 'test', 'test_full.csv')

# 输出：最终验证结果目录
OUTPUT_DIR = os.path.join(BASE_DIR, 'results', 'final_validation', 'b6')
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PRED = os.path.join(OUTPUT_DIR, 'predictions.csv')

# 推理参数
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEQ_LENGTH = 128  # 与训练时一致
# ====================================================

def predict():
    print("=== B6 RoBERTa-large SimCSE - 最终测试集推理 ===")
    print(f"使用设备: {DEVICE}")
    if DEVICE == "cuda":
        print(f"显卡: {torch.cuda.get_device_name(0)}")
    
    # 1. 加载预训练模型（SimCSE微调后）
    model_path = os.path.join(PRETRAIN_DIR, MODEL_SUBDIR)
    if not os.path.exists(model_path):
        # 兼容：如果找不到final_model子目录，尝试直接加载roberta_simcse目录
        model_path = os.path.join(PRETRAIN_DIR, "roberta_simcse")
    
    print(f"加载模型: {model_path}")
    try:
        model = SentenceTransformer(model_path)
        model.max_seq_length = MAX_SEQ_LENGTH
        model.to(DEVICE)
        print(f"模型加载成功，参数量: ~355M")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请确认训练脚本已成功运行并保存模型到 results/b6/")
        return
    
    # 2. 加载预计算的风格锚点（中心向量）
    center_path = os.path.join(PRETRAIN_DIR, CENTER_VECTOR_FILE)
    if not os.path.exists(center_path):
        print(f"错误: 中心向量文件未找到: {center_path}")
        print("请先运行训练脚本生成中心向量")
        return
    
    print(f"加载中心向量: {center_path}")
    center = np.load(center_path)
    center_tensor = torch.from_numpy(center).to(DEVICE)
    print(f"锚点向量维度: {center.shape[0]}")
    
    # 3. 读取最终测试集（210句，含ID/OOD）
    test_df = pd.read_csv(TEST_PATH)
    if 'text' not in test_df.columns:
        print(f"错误: {TEST_PATH} 缺少'text'列")
        return
        
    texts = test_df['text'].astype(str).tolist()
    print(f"测试集规模: {len(texts)}句")
    
    # 4. 批量编码测试集
    print("编码测试集...")
    embeddings = model.encode(
        texts, 
        convert_to_tensor=True, 
        device=DEVICE,
        batch_size=BATCH_SIZE,
        show_progress_bar=True
    )
    
    # 5. 计算与风格锚点的偏离度（0-10分）
    similarities = F.cosine_similarity(embeddings, center_tensor.unsqueeze(0), dim=1)
    sim_scores = similarities.cpu().numpy()
    
    # 映射到0-10分（与项目其他基线保持一致）
    # 原预实验: (sim + 1) / 2 * 100 -> 0-100
    # 现统一为: (sim + 1) / 2 * 10 -> 0-10
    scores = (sim_scores + 1) / 2 * 10
    
    # 6. 保存结果（等待人类基准对齐计算Spearman）
    results = pd.DataFrame({
        'text_id': test_df.get('text_id', range(len(texts))),
        'text': texts,
        'b6_score': scores,
        'cosine_sim_raw': sim_scores  # 保留原始[-1,1]范围相似度
    })
    results.to_csv(OUTPUT_PRED, index=False, encoding='utf-8-sig')
    
    print(f"\n结果已保存: {OUTPUT_PRED}")
    print(f"分数统计: min={scores.min():.2f}, max={scores.max():.2f}, mean={scores.mean():.2f}, std={scores.std():.2f}")
    print(f"待人类基准打分完成后，合并计算Spearman相关性")

if __name__ == "__main__":
    predict()

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