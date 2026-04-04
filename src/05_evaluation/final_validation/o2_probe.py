"""
O2_probe: MiniLM SimCSE (仅投影头可训练) - 最终测试集推理
加载预实验训练的轻量投影头模型 (384->128维)，直接计算风格偏离度
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path
import torch.nn.functional as F

# ---------- 添加的显存限制代码 ----------
if os.getenv('CUDA_MEMORY_FRACTION'):
    torch.cuda.set_per_process_memory_fraction(float(os.getenv('CUDA_MEMORY_FRACTION')))
    print(f"显存限制: {float(os.getenv('CUDA_MEMORY_FRACTION'))*100:.0f}%")
# -------------------------------------

# ==================== 参数配置区 ====================
# 路径配置（4层parent适配final_validation路径）
BASE_DIR = Path(__file__).resolve().parents[3]

# 输入：预实验训练的O2_probe模型和锚点（只读）
PRETRAIN_DIR = BASE_DIR / "results" / "o2_probe"
MODEL_SUBDIR = "simcse_probe/final_model"  # 包含投影头的SimCSE模型
CENTER_VECTOR_FILE = "center_vector.npy"

# 输入：最终测试集（210句完整测试集）
TEST_PATH = BASE_DIR / "data" / "test" / "test_full.csv"

# 输出：最终验证结果目录
OUTPUT_DIR = BASE_DIR / "results" / "final_validation" / "o2_probe"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PRED = OUTPUT_DIR / "predictions.csv"

# 推理参数
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ====================================================

def predict():
    print("=== O2_probe MiniLM+投影头 SimCSE - 最终测试集推理 ===")
    print(f"使用设备: {DEVICE}")
    if DEVICE == "cuda":
        print(f"显卡: {torch.cuda.get_device_name(0)}")

    # 1. 加载预训练模型（MiniLM + 128维投影头）
    model_path = PRETRAIN_DIR / MODEL_SUBDIR
    print(f"加载模型: {model_path}")
    try:
        model = SentenceTransformer(str(model_path))
        model.to(DEVICE)
        # 验证投影头维度（应为128）
        print(f"模型加载成功，输出维度: {model.get_sentence_embedding_dimension()}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请确认训练脚本已成功运行并保存模型到 results/o2_probe/simcse_probe/")
        return

    # 2. 加载预计算的风格锚点（中心向量，128维）
    center_path = PRETRAIN_DIR / CENTER_VECTOR_FILE
    print(f"加载中心向量: {center_path}")
    if not center_path.exists():
        print("错误: 中心向量文件未找到，请先运行训练脚本")
        return
    
    center = np.load(center_path)
    center_tensor = torch.from_numpy(center).to(DEVICE)
    print(f"锚点向量维度: {center.shape[0]} (应为128)")

    # 3. 读取最终测试集（210句，含ID/OOD）
    print(f"读取测试集: {TEST_PATH}")
    test_df = pd.read_csv(TEST_PATH)
    if 'text' not in test_df.columns:
        print("错误: 测试集缺少'text'列")
        return
        
    texts = test_df['text'].astype(str).tolist()
    print(f"测试集规模: {len(texts)}句")

    # 4. 批量编码测试集（通过投影头得到128维向量）
    print("编码测试集...")
    embeddings = model.encode(
        texts, 
        convert_to_tensor=True, 
        device=DEVICE,
        batch_size=BATCH_SIZE,
        show_progress_bar=True
    )
    
    # 验证维度
    print(f"测试集嵌入维度: {embeddings.shape[1]}")

    # 5. 计算与风格锚点的偏离度（0-10分）
    similarities = F.cosine_similarity(embeddings, center_tensor.unsqueeze(0), dim=1)
    sim_scores = similarities.cpu().numpy()
    
    # 映射到0-10分（与项目其他基线保持一致）
    scores = (sim_scores + 1) / 2 * 10  # [-1,1] -> [0,10]

    # 6. 保存结果（等待人类基准对齐计算Spearman）
    results = pd.DataFrame({
        'text_id': test_df.get('text_id', range(len(texts))),
        'text': texts,
        'o2_probe_score': scores,
        'cosine_sim_raw': sim_scores  # 保留原始[-1,1]范围相似度
    })
    results.to_csv(OUTPUT_PRED, index=False, encoding='utf-8-sig')

    print(f"\n结果已保存: {OUTPUT_PRED}")
    print(f"分数统计: min={scores.min():.2f}, max={scores.max():.2f}, mean={scores.mean():.2f}, std={scores.std():.2f}")
    print(f"待人类基准打分完成后，合并计算Spearman相关性")

if __name__ == "__main__":
    predict()

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