import pandas as pd

"""
prepare_ablation.py
根据测试集ID从原始语料库提取原文，生成 test_ablation.csv 用于消融实验
历史遗留：语料库ID为1-based（从1开始），与测试集ID一致
"""

# 1. 读取测试集（含id列，1-based）
test_df = pd.read_csv('data/test/test_positive.csv')

# 2. 读取原语料库，显式创建1-based ID（与历史文件保持一致）
raw_df = pd.read_csv('data/processed/logos_cleaned.csv')
raw_df['id'] = range(1, len(raw_df) + 1)  # 1,2,3... 与语料库实际行号一致

# 3. 数值匹配（merge不关心物理位置，只关心id数值相等）
ablation_df = test_df[['id']].merge(
    raw_df[['id', 'text']], 
    on='id', 
    how='left'
)

# 4. 验证（抽查）
sample_id = test_df['id'].iloc[0]
print(f"验证ID={sample_id}:")
print(f"  测试集（脱敏）: {test_df[test_df['id']==sample_id]['text'].values[0][:30]}...")
print(f"  语料库（原文）: {ablation_df[ablation_df['id']==sample_id]['text'].values[0][:30]}...")

# 5. 保存（格式与测试集一致：id,text）
ablation_df.to_csv('data/test/test_ablation.csv', index=False, encoding='utf-8-sig')
print(f"\n已生成 test_ablation.csv：{len(ablation_df)} 句")
