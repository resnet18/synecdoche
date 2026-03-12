#!/usr/bin/env python3
"""
游戏外OOD转换器 - 放在 ./src/02_preprocessing/txt_to_csv.py
运行方式（从项目根目录）：
    python src/02_preprocessing/txt_to_csv.py
"""

import pandas as pd
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='将游戏外OOD文本(txt)转换为标准CSV')
    parser.add_argument('--input', '-i', 
                       default='./data/raw/outgame.txt',      # 数据位置
                       help='输入txt路径 (默认: ./data/raw/outgame.txt)')
    parser.add_argument('--output', '-o',
                       default='./data/test/ood_outgame.csv',  # 输出位置
                       help='输出CSV路径 (默认: ./data/test/ood_outgame.csv)')
    return parser.parse_args()

def main():
    args = parse_args()
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"[!] 错误：找不到 {input_path}")
        print("提示：请确保在项目根目录运行，或检查文件路径")
        return

    texts, lengths, categories = [], [], []
    current_category = "未分类"
    
    # 读取并自动解析类别（以#开头的行为类别标记）
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 检测类别标记：# 哲学宣言（...） -> 提取"哲学"
            if line.startswith('#'):
                # 去掉#和空格，提取"（"前的第一部分作为类别名
                category_raw = line.lstrip('#').strip()
                category = category_raw.split('（')[0].split()[0]
                if category:
                    current_category = category
                continue
            
            texts.append(line)
            lengths.append(len(line))
            categories.append(current_category)
    
    # 生成标准格式CSV
    df = pd.DataFrame({
        'text': texts,
        'length': lengths,
        'label': 0,                      # OOD统一标签
        'ood_type': 'out_game',          # 标记为游戏外
        'source_character': categories   # 用类别代替角色名（如"哲学"）
    })
    
    # 确保输出目录存在（自动创建）
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"完成：{len(df)}句 -> {output_path}")
    print(f"类别分布：\n{df['source_character'].value_counts()}")

if __name__ == "__main__":
    main()
