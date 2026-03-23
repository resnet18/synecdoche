import pandas as pd
import numpy as np
import joblib
import re
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import os

# ========== 路径配置 ===========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
BASE_DIR = os.path.join(PROJECT_ROOT, 'results')
TEST_ORIG_PATH = os.path.join(PROJECT_ROOT, 'data', 'test', 'test_ablation.csv')
ABLATION_DIR = os.path.join(BASE_DIR, 'ablation')

os.makedirs(ABLATION_DIR, exist_ok=True)

print(f"项目根目录: {PROJECT_ROOT}")
print(f"消融输出: {ABLATION_DIR}")
print("=" * 60)

# ========== 函数 ===========
def load_test():
    df = pd.read_csv(TEST_ORIG_PATH)
    print(f"加载测试集: {len(df)} 句")
    return df['id'].values, df['text'].values

def punct_features(texts, ref_lengths=None):
    """提取标点特征，支持外部参考长度（用于对齐原文与脱敏文本）"""
    features = []
    for i, text in enumerate(texts):
        # 去除中括号占位符（脱敏文本），保留自然标点和文本
        clean_text = re.sub(r'\[[^\]]+\]', '', text)
        
        # === 关键修复：使用参考长度（原文长度）作为分母 ===
        length = ref_lengths[i] if ref_lengths is not None else max(len(clean_text), 1)
        
        # 统计clean_text中的标点（脱敏前后标点应该一致）
        ellipsis = len(re.findall(r'…+|\.{3,}', clean_text))
        dash = len(re.findall(r'——|--+', clean_text))
        comma = len(re.findall(r'，|,', clean_text))
        period = len(re.findall(r'。|\.', clean_text))
        question = len(re.findall(r'？|\?', clean_text))
        exclaim = len(re.findall(r'！|!', clean_text))
        
        features.append([ellipsis/length, dash/length, comma/length,
                        period/length, question/length, exclaim/length])
    return np.array(features)

def calc_method(base_path, texts, name, ref_lengths=None):
    """计算分数：B2用长度对齐，其他基线用vectorizer"""
    if name == 'B2':
        center = np.load(os.path.join(base_path, 'center_vector.npy'))
        X = punct_features(texts, ref_lengths)
        scores = cosine_similarity(X, center.reshape(1,-1)).flatten()
    else:
        # B0/B1/B3: 词袋/n-gram/TF-IDF，不需要长度对齐（不是密度特征）
        vec = joblib.load(os.path.join(base_path, 'vectorizer.pkl'))
        center = np.load(os.path.join(base_path, 'center_vector.npy'))
        X = vec.transform(texts)
        scores = cosine_similarity(X, center.reshape(1,-1)).flatten()
    
    if name == 'B1':
            print(f"  [B1调试] 分数均值={scores.mean():.4f}, 方差={scores.var():.6f}, 唯一值={len(set(scores))}")
            # 如果方差极小，打印前3个样本的特征向量非零元素数
            if scores.var() < 1e-10:
                print(f"  [B1警告] 所有样本得分几乎相同，检查vectorizer词汇表大小: {len(vec.vocabulary_)}")

    print(f"{name}: 分数范围 [{scores.min():.3f}, {scores.max():.3f}]")
    return scores

# ========== 主程序（只有一个main） ===========
def main():
    ids, texts = load_test()  # 加载原文（test_ablation.csv）
    
    # === 预计算原文长度（作为B2的统一基准） ===
    ref_lengths = [len(re.sub(r'\[[^\]]+\]', '', t)) for t in texts]
    
    summary = []

    for method in ['b0', 'b1', 'b2', 'b3']:
        print(f"\n处理 {method.upper()}...")
        base = os.path.join(BASE_DIR, method)

        # 1. 计算原文分数（B2会用到ref_lengths，其他基线忽略）
        scores_orig = calc_method(base, texts, method.upper(), ref_lengths=ref_lengths)

        # 2. 保存原文分数（覆盖之前的错误文件）
        df_orig = pd.DataFrame({'id': ids, 'text': texts, 'score': scores_orig})
        df_orig.to_csv(os.path.join(base, 'predictions_original.csv'), index=False, encoding='utf-8-sig')

        # 3. 加载脱敏文本并计算分数
        des_path = os.path.join(base, 'predictions.csv')
        if not os.path.exists(des_path):
            print(f"  警告：{des_path} 不存在，跳过")
            continue
            
        des_df = pd.read_csv(des_path)
        des_texts = des_df['text'].values
        
        # === 关键：B2使用同样的ref_lengths，确保长度基准一致 ===
        scores_des = calc_method(base, des_texts, method.upper(), 
                                ref_lengths=ref_lengths if method == 'b2' else None)
        
        # 4. 计算Spearman（现在B2的尺度一致了）
        rho, p = spearmanr(scores_des, scores_orig)
        print(f"  {method.upper()} Spearman ρ = {rho:.4f}, p = {p:.4f}")
        summary.append({
            'method': method.upper(),
            'spearman_rho': rho,
            'p_value': p
        })

    # 保存汇总
    if summary:
        summary_df = pd.DataFrame(summary)
        summary_path = os.path.join(ABLATION_DIR, 'consistency_summary.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n总结已保存至: {summary_path}")
        print(summary_df)

if __name__ == "__main__":
    main()
    print("\n跑完了，去睡觉！")
