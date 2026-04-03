"""
O1a: O1-v4 输入消融实验（原文 vs 脱敏一致性检验）
同步修改：StandardScaler + 欧氏距离（与主脚本一致）
"""

import pandas as pd
import numpy as np
import re
import jieba
import pickle
import os
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

# ========== 路径（与模板一致）==========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
BASE_DIR = os.path.join(PROJECT_ROOT, 'results', 'o1')
ABLATION_DIR = os.path.join(PROJECT_ROOT, 'results', 'ablation')
TEST_ORIG_PATH = os.path.join(PROJECT_ROOT, 'data', 'test', 'test_ablation.csv')

os.makedirs(ABLATION_DIR, exist_ok=True)

print(f"项目根目录: {PROJECT_ROOT}")
print(f"O1目录: {BASE_DIR}")
print(f"消融输出: {ABLATION_DIR}")
print("=" * 60)

# ========== 1. 语域词典（确保在函数定义之前）==========
LITERARY_PARTICLES = {
    '之', '其', '而', '以', '于', '为', '与', '乃', '所', '被', '把', '且',
    '并', '却', '但', '然', '则', '故', '因', '虽', '若', '苟', '纵', '即',
    '已', '亦', '皆', '尽', '俱', '均', '咸', '悉', '皆尽', '全都',
    '尚', '仍', '还', '依然', '依旧', '仍然', '仍旧',
    '必', '定', '须', '应', '当', '该', '须得', '定要', '必将',
    '勿', '毋', '莫', '不', '未', '无', '非', '勿要', '不要', '不必',
    '甚', '颇', '极', '甚为', '很是', '相当', '非常',
    '何', '胡', '焉', '安', '孰', '谁', '何以', '为何',
    '即', '便', '就', '系', '是', '非', '即是', '便是', '就是',
    '仅', '只', '唯', '惟', '只是', '仅仅', '只不过'
}

MODAL_MARKERS = {
    '无妨', '不妨', '可否', '能否', '是否',
    '恐怕', '怕是', '唯恐', '生怕', '大概', '大约', '似乎', '好像',
    '应当', '应该', '理当', '该当', '自当', '本当',
    '未必', '不一定', '不见得', '难说', '未知', '未可知',
    '终究', '终将', '毕竟', '到底', '究竟', '终归', '总归', '迟早',
    '已然', '已经', '早已', '业已', '皆已', '俱已'
}

LOGICAL_CONNECTORS = {
    '因此', '所以', '因而', '从而', '于是', '因为', '由于', '缘于', '鉴于',
    '但是', '然而', '不过', '只是', '固然', '尽管', '虽然',
    '如果', '若', '若是', '假使', '假如', '倘若', '只要', '除非', '无论',
    '而且', '并且', '况且', '何况', '甚至', '乃至', '不仅', '不只', '不但'
}

FUNCTION_WORDS = LITERARY_PARTICLES | MODAL_MARKERS | LOGICAL_CONNECTORS

# ========== 2. 特征提取（9维零标签依赖）==========
def extract_o1_features(texts):
    """提取 O1-v4 的 9 维纯文本特征"""
    features = []
    for text in texts:
        if not isinstance(text, str) or not text:
            features.append(np.zeros(9))
            continue
        
        char_count = max(len(text), 1)
        words = list(jieba.cut(text))
        word_count = max(len(words), 1)
        
        sentences = [s.strip() for s in re.split(r'[。！？;；!?]', text) if s.strip()]
        sent_count = max(len(sentences), 1)
        
        # 语域标记密度（现在 FUNCTION_WORDS 已经在全局定义）
        func_count = sum(1 for w in words if w in FUNCTION_WORDS)
        func_density = func_count / word_count
        
        literary_count = sum(1 for w in words if w in LITERARY_PARTICLES)
        conn_count = sum(1 for w in words if w in LOGICAL_CONNECTORS)
        literary_density = literary_count / word_count
        connector_density = conn_count / word_count
        
        # 句法复杂度
        avg_sent_len = char_count / sent_count
        internal_pauses = len(re.findall(r'[，、；,;]', text))
        clause_density = internal_pauses / sent_count
        
        if len(sentences) > 1:
            sent_lens = [len(s) for s in sentences]
            cv_sent_len = np.std(sent_lens) / np.mean(sent_lens) if np.mean(sent_lens) > 0 else 0
        else:
            cv_sent_len = 0
        
        # 标点指纹
        ellipsis_count = len(re.findall(r'…+|\.{3,}', text))
        ellipsis_density = ellipsis_count / char_count
        emotion_count = text.count('？') + text.count('！') + text.count('?') + text.count('!')
        emotion_density = emotion_count / char_count
        ultra_short = sum(1 for s in sentences if len(s) <= 5)
        ultra_short_ratio = ultra_short / sent_count
        
        features.append([
            func_density, literary_density, connector_density,
            avg_sent_len, clause_density, cv_sent_len,
            ellipsis_density, emotion_density, ultra_short_ratio
        ])
    
    return np.array(features)

# ========== 3. 主程序（同步 O1-v4 逻辑）==========
def main():
    # 加载原文测试集
    df_test = pd.read_csv(TEST_ORIG_PATH)
    texts = df_test['text'].astype(str).tolist()
    print(f"加载原始测试集: {len(texts)} 句")
    
    # 加载 StandardScaler 和欧氏版中心向量
    scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
    center_path = os.path.join(BASE_DIR, 'center_vector.npy')
    
    if not os.path.exists(scaler_path):
        print(f"错误: 未找到 scaler {scaler_path}，请先运行 O1-v4 主脚本")
        exit()
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    center_vector = np.load(center_path)
    print(f"加载 StandardScaler 和中心向量: {center_vector.shape[0]} 维")
    
    # 提取特征
    print("编码原始文本...")
    X_orig = extract_o1_features(texts)
    
    # StandardScaler 变换（用训练集的 scaler）
    X_orig_scaled = scaler.transform(X_orig)
    
    # 欧氏距离 + 距离转 0-100 分
    distances = euclidean_distances(X_orig_scaled, center_vector.reshape(1, -1)).flatten()
    max_dist = distances.max() if distances.max() > 0 else 1
    scores = (1 - distances / max_dist) * 100
    
    # 保存原文预测结果
    df_orig = pd.DataFrame({
        'text': texts,
        'o1_score': scores,
        'euclidean_distance': distances
    })
    df_orig.to_csv(os.path.join(BASE_DIR, 'predictions_original.csv'), 
                   index=False, encoding='utf-8-sig')
    print(f"原始分数: [{scores.min():.3f}, {scores.max():.3f}]")
    
    # 对比脱敏版本
    des_path = os.path.join(BASE_DIR, 'predictions.csv')
    if not os.path.exists(des_path):
        print(f"错误：找不到脱敏结果 {des_path}")
        exit()
    
    df_des = pd.read_csv(des_path)
    des_scores = df_des['o1_score'].values
    
    # 确保长度一致
    if len(des_scores) != len(scores):
        print(f"警告: 样本数不一致（脱敏{len(des_scores)} vs 原文{len(scores)}）")
        min_len = min(len(des_scores), len(scores))
        des_scores = des_scores[:min_len]
        scores = scores[:min_len]
        print(f"  已截断至前{min_len}个样本")
    
    # 计算 Spearman
    rho, p = spearmanr(des_scores, scores)
    print(f"\nO1 Spearman ρ = {rho:.4f} (p={p:.4f})")
    
    # 追加到汇总表
    summary_path = os.path.join(ABLATION_DIR, 'consistency_summary.csv')
    
    new_record = {
        'method': 'O1',
        'spearman_rho': round(rho, 4),
        'p_value': round(p, 4)
    }
    
    try:
        if os.path.exists(summary_path):
            df_summary = pd.read_csv(summary_path)
            df_summary = df_summary[df_summary['method'] != 'O1']
        else:
            df_summary = pd.DataFrame(columns=['method', 'spearman_rho', 'p_value'])
        
        df_summary = pd.concat([df_summary, pd.DataFrame([new_record])], ignore_index=True)
        df_summary = df_summary.sort_values('method').reset_index(drop=True)
        
        df_summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n已更新汇总表: {summary_path}")
        print(df_summary)
        
    except Exception as e:
        print(f"\n警告：无法更新汇总表: {e}")
        print(f"手动记录: O1, ρ={rho:.4f}, p={p:.4f}")
    
    # 诊断
    if rho > 0.9:
        print("\n强一致：O1 内容-风格解耦成功")
    elif rho > 0.7:
        print("\n中等一致：部分受内容影响")
    else:
        print("\n弱一致：存在内容泄漏")

    print("=== 完成 ===")

if __name__ == "__main__":
    main()