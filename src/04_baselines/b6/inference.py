"""
B6 推理: Chinese-RoBERTa-large SimCSE
使用训练好的模型计算句子与中心向量的余弦相似度
支持批处理，所有参数前置
"""

import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path
import torch.nn.functional as F

# ==================== 参数配置区 ====================
# 路径配置（应与训练脚本一致）
PROJECT_ROOT = Path(__file__).resolve().parents[3]      # 向上回溯3层到项目根目录
DATA_DIR = PROJECT_ROOT / "data" / "test"               # 测试集所在目录
OUTPUT_DIR = PROJECT_ROOT / "results" / "b6"            # 结果输出目录

# 模型配置
MODEL_SUBDIR = "roberta_simcse/final_model"             # 训练好的模型子目录（注意：Trainer保存的是final_model子目录）
CENTER_VECTOR_FILE = "center_vector.npy"                # 中心向量文件名

# 推理参数
BATCH_SIZE = 16                                         # 推理batch size（355M模型，16G显存可设16-32）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # 自动选择设备
MAP_TO_100 = True                                       # 是否将相似度 [-1,1] 映射到 [0,100]
MAX_SEQ_LENGTH = 128                                    # 应与训练时一致
# ====================================================

def predict():
    print(f"使用设备: {DEVICE}")
    if DEVICE == "cuda":
        print(f"显卡: {torch.cuda.get_device_name(0)}")
    
    # 加载模型（中文RoBERTa-large SimCSE）
    model_path = OUTPUT_DIR / MODEL_SUBDIR
    if not model_path.exists():
        # 如果找不到final_model子目录，尝试直接加载roberta_simcse目录
        model_path = OUTPUT_DIR / "roberta_simcse"
    
    print(f"加载模型: {model_path}")
    try:
        model = SentenceTransformer(str(model_path))
        model.max_seq_length = MAX_SEQ_LENGTH  # 确保与训练时一致
        model.to(DEVICE)
        print(f"模型加载成功，参数量: ~355M")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请确认训练脚本已成功运行并保存模型")
        return
    
    # 加载中心向量
    center_path = OUTPUT_DIR / CENTER_VECTOR_FILE
    if not center_path.exists():
        print(f"错误: 中心向量文件未找到: {center_path}")
        print("请先运行训练脚本生成中心向量")
        return
    
    print(f"加载中心向量: {center_path}")
    center = np.load(center_path)
    center_tensor = torch.tensor(center).to(DEVICE)
    print(f"中心向量维度: {center.shape[0]}")
    
    # 加载测试集（支持多个测试文件）
    test_files = ["test_positive.csv"]
    for test_file in test_files:
        test_path = DATA_DIR / test_file
        if not test_path.exists():
            print(f"警告: 未找到测试文件 {test_file}，跳过")
            continue
            
        test_df = pd.read_csv(test_path)
        if 'text' not in test_df.columns:
            print(f"错误: {test_file} 缺少'text'列")
            continue
            
        texts = test_df['text'].tolist()
        print(f"\n处理测试集: {test_file} ({len(texts)} 条样本)")
        
        # 批量编码
        print("开始推理...")
        embeddings = model.encode(
            texts, 
            convert_to_tensor=True, 
            device=DEVICE,
            batch_size=BATCH_SIZE,
            show_progress_bar=True
        )
        
        # 计算余弦相似度
        similarities = F.cosine_similarity(embeddings, center_tensor.unsqueeze(0), dim=1)
        sim_scores = similarities.cpu().numpy()
        
        # 映射到 0-100
        if MAP_TO_100:
            scores = (sim_scores + 1) / 2 * 100   # [-1,1] -> [0,100]
        else:
            scores = sim_scores
        
        # 保存结果
        result_df = pd.DataFrame({
            'text': texts,
            'score': scores
        })
        
        # 根据输入文件名生成输出文件名
        out_filename = f"predictions.csv"
        out_path = OUTPUT_DIR / out_filename
        result_df.to_csv(out_path, index=False)
        
        print(f"结果已保存: {out_path}")
        print(f"分数统计: min={scores.min():.2f}, max={scores.max():.2f}, mean={scores.mean():.2f}")

    print("\n✓ 推理完成")

if __name__ == "__main__":
    predict()