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

def punct_features(texts):
    features = []
    for text in texts:
        length = len(text)
        if length == 0:
            features.append([0]*6)
            continue
        ellipsis = len(re.findall(r'…+|\.{3,}', text))
        dash = len(re.findall(r'——|--+', text))
        comma = len(re.findall(r'，|,', text))
        period = len(re.findall(r'。|\.', text))
        question = len(re.findall(r'？|\?', text))
        exclaim = len(re.findall(r'！|!', text))
        features.append([ellipsis/length, dash/length, comma/length,
                        period/length, question/length, exclaim/length])
    return np.array(features)

def calc_method(base_path, texts, name):
    if name == 'B2':
        center = np.load(os.path.join(base_path, 'center_vector.npy'))
        X = punct_features(texts)
    else:
        vec = joblib.load(os.path.join(base_path, 'vectorizer.pkl'))
        center = np.load(os.path.join(base_path, 'center_vector.npy'))
        X = vec.transform(texts)
    scores = cosine_similarity(X, center.reshape(1,-1)).flatten()
    print(f"{name}: 分数范围 [{scores.min():.3f}, {scores.max():.3f}]")
    return scores

# ========== 主程序 ===========
def main():
    ids, texts = load_test()
    summary = []   # 用于收集每个方法的相关系数

    for method in ['b0', 'b1', 'b2', 'b3']:
        print(f"\n处理 {method.upper()}...")
        base = os.path.join(BASE_DIR, method)

        # 1. 计算原始文本的分数（即 ablation 方法自己的打分）
        scores = calc_method(base, texts, method.upper())

        # 2. 保存为 predictions_original.csv（以备后用）
        df_orig = pd.DataFrame({'id': ids, 'text': texts, 'score': scores})
        orig_save_path = os.path.join(base, 'predictions_original.csv')
        df_orig.to_csv(orig_save_path, index=False, encoding='utf-8-sig')

        # 3. 对比脱敏版本的分数（假设已存在 predictions.csv）
        des_path = os.path.join(base, 'predictions.csv')
        if os.path.exists(des_path):
            des_df = pd.read_csv(des_path)

            # 尝试按列名获取分数，优先使用 'similarity_raw' 或 'b0_score'（取决于你的实际列名）
            # 如果找不到，则回退到第二列（兼容旧格式）
            score_col_candidates = ['similarity_raw', f'{method}_score', 'score']
            found_col = None
            for col in score_col_candidates:
                if col in des_df.columns:
                    found_col = col
                    break
            if found_col is not None:
                des_scores = des_df[found_col].values
            else:
                # 如果都没有，就取第二列（假设第一列是文本或ID）
                des_scores = des_df.iloc[:, 1].values
                print(f"  未找到标准列名，使用第二列作为分数。")

            # 计算 Spearman 相关系数（假设行顺序一致）
            rho, p = spearmanr(des_scores, df_orig['score'].values)
            print(f"  {method.upper()} Spearman ρ = {rho:.4f}, p = {p:.4f}")

            # 将结果加入 summary
            summary.append({
                'method': method.upper(),
                'spearman_rho': rho,
                'p_value': p
            })
        else:
            print(f"  警告：{des_path} 不存在，跳过相关系数计算")

    # ===== 保存汇总结果 =====
    if summary:
        summary_df = pd.DataFrame(summary)
        summary_path = os.path.join(ABLATION_DIR, 'consistency_summary.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n总结已保存至: {summary_path}")
        print(summary_df)
    else:
        print("\n未收集到任何相关系数，请检查 predictions.csv 是否存在或列名是否正确。")

if __name__ == "__main__":
    main()
    print("\n跑完了，去睡觉！")