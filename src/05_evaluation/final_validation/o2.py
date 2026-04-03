"""
O2: MiniLM SimCSE微调模型 - 最终测试集推理
使用预实验训练好的SimCSE模型，直接加载中心向量，零重新训练
"""

import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path
import torch.nn.functional as F
import os

# ==================== 参数配置区 ====================
# 路径配置（4层parent适配final_validation路径）
BASE_DIR = Path(__file__).resolve().parents[3]

# 输入：预实验训练好的SimCSE模型和锚点（只读）
PRETRAIN_DIR = BASE_DIR / "results" / "o2"
MODEL_SUBDIR = "simcse_model/final_model"  # 训练好的SimCSE模型子目录
CENTER_VECTOR_FILE = "center_vector.npy"

# 输入：最终测试集（210句完整测试集）
TEST_PATH = BASE_DIR / "data" / "test" / "test_full.csv"

# 输出：最终验证结果目录
OUTPUT_DIR = BASE_DIR / "results" / "final_validation" / "o2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PRED = OUTPUT_DIR / "predictions.csv"

# 推理参数
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ====================================================

def predict():
    print("=== O2 MiniLM SimCSE - 最终测试集推理 ===")
    print(f"使用设备: {DEVICE}")
    if DEVICE == "cuda":
        print(f"显卡: {torch.cuda.get_device_name(0)}")
    
    # 1. 加载预训练SimCSE模型（MiniLM+SimCSE微调）
    model_path = PRETRAIN_DIR / MODEL_SUBDIR
    print(f"加载模型: {model_path}")
    try:
        model = SentenceTransformer(str(model_path))
        model.to(DEVICE)
        print("SimCSE模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请确认训练脚本已成功运行并保存模型到 results/o2/simcse_model/")
        return
    
    # 2. 加载预计算的风格锚点（中心向量）
    center_path = PRETRAIN_DIR / CENTER_VECTOR_FILE
    print(f"加载中心向量: {center_path}")
    if not center_path.exists():
        print("错误: 中心向量文件未找到，请先运行训练脚本")
        return
    
    center = np.load(center_path)
    center_tensor = torch.from_numpy(center).to(DEVICE)
    print(f"锚点向量维度: {center.shape[0]}")
    
    # 3. 读取最终测试集（210句，含ID/OOD）
    print(f"读取测试集: {TEST_PATH}")
    test_df = pd.read_csv(TEST_PATH)
    if 'text' not in test_df.columns:
        print("错误: 测试集缺少'text'列")
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
    # 原预实验可能使用0-100，现统一为0-10
    scores = (sim_scores + 1) / 2 * 10  # [-1,1] -> [0,10]
    
    # 6. 保存结果（等待人类基准对齐计算Spearman）
    results = pd.DataFrame({
        'text_id': test_df.get('text_id', range(len(texts))),
        'text': texts,
        'o2_score': scores,
        'cosine_sim_raw': sim_scores  # 保留原始[-1,1]范围相似度
    })
    results.to_csv(OUTPUT_PRED, index=False, encoding='utf-8-sig')
    
    print(f"\n结果已保存: {OUTPUT_PRED}")
    print(f"分数统计: min={scores.min():.2f}, max={scores.max():.2f}, mean={scores.mean():.2f}, std={scores.std():.2f}")
    print(f"待人类基准打分完成后，合并计算Spearman相关性")

if __name__ == "__main__":
    predict()