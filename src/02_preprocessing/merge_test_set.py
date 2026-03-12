#!/usr/bin/env python3
"""
合并测试集脚本（统一列格式）
用法：python src/02_preprocessing/merge_test_set.py
"""

import pandas as pd
import argparse
from pathlib import Path

# === 配置（修改这里）===
TEST_POS_PATH = './data/test/test_positive.csv'           # 现有的70句正例
IN_GAME_PATH = './data/test/game_ood_ready.csv'           # 游戏内OOD（你的脚本输出）
OUT_GAME_PATH = './data/test/ood_outgame.csv'             # 游戏外OOD（txt转的）
OUTPUT_DIR = './data/test'
LONG_THRESHOLD = 35                                       # 与划分脚本一致

def main():
    # 1. 读取正例（保持原样）
    pos_df = pd.read_csv(TEST_POS_PATH)
    print(f"[√] 正例：{len(pos_df)}句")
    
    # 2. 读取并合并OOD
    ood_frames = []
    
    # 游戏内OOD：列名 text, length, label, ood_type, source_character
    if Path(IN_GAME_PATH).exists():
        df_in = pd.read_csv(IN_GAME_PATH)
        df_in = df_in.rename(columns={'source_character': 'source'})
        ood_frames.append(df_in)
        print(f"[√] 游戏内OOD：{len(df_in)}句")
    
    # 游戏外OOD：列名相同
    if Path(OUT_GAME_PATH).exists():
        df_out = pd.read_csv(OUT_GAME_PATH)
        df_out = df_out.rename(columns={'source_character': 'source'})
        ood_frames.append(df_out)
        print(f"[√] 游戏外OOD：{len(df_out)}句")
    
    if not ood_frames:
        print("[!] 没有找到OOD文件")
        return
        
    ood_df = pd.concat(ood_frames, ignore_index=True)
    
    # 3. 格式转换（对齐到test_positive的列结构）
    # 生成负id（从-1开始）
    ood_df['id'] = [-i for i in range(1, len(ood_df)+1)]
    
    # 计算is_long（基于length > 35）
    ood_df['is_long'] = ood_df['length'] > LONG_THRESHOLD
    
    # 确保label=0
    ood_df['label'] = 0
    
    # 选择并排列列（与test_positive一致：id, text, length, is_long, label, source）
    ood_final = ood_df[['id', 'text', 'length', 'is_long', 'label', 'source']]
    
    # 4. 保存纯OOD文件（替换原来的test_negative_template）
    neg_path = Path(OUTPUT_DIR) / 'test_negative.csv'
    ood_final.to_csv(neg_path, index=False, encoding='utf-8-sig')
    print(f"[√] 生成：{neg_path}（{len(ood_final)}句）")
    
    # 5. 合并完整测试集（正例+OOD）
    # 保持正例在前或打乱？建议打乱避免顺序偏差
    full_df = pd.concat([pos_df, ood_final], ignore_index=True)
    full_df = full_df.sample(frac=1, random_state=325).reset_index(drop=True)  # 325种子
    
    full_path = Path(OUTPUT_DIR) / 'test_full.csv'
    full_df.to_csv(full_path, index=False, encoding='utf-8-sig')
    print(f"[√] 生成：{full_path}（总计{len(full_df)}句）")
    
    # 6. 统计信息
    print(f"\n构成统计：")
    print(f"  ID（Logos）：{len(pos_df)}句")
    print(f"  OOD合计：{len(ood_final)}句")
    for src, count in ood_final['source'].value_counts().items():
        print(f"    - {src}: {count}句")
    print(f"\n长度分布：")
    print(f"  长句(>{LONG_THRESHOLD}字): {full_df['is_long'].sum()}句")
    print(f"  短句(<={LONG_THRESHOLD}字): {(~full_df['is_long']).sum()}句")

if __name__ == "__main__":
    main()
