"""
B8 Ablation: 原始文本 vs 脱敏文本一致性检验
关键：参考文本固定（reference.csv），根据显存决定15/5/0句，仅测试文本变化
"""
import os
import sys
from pathlib import Path

# 关键：向上3层到项目根（因为b8a.py在src/05_evaluation/ablation/）
BASE_DIR = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, BASE_DIR)

from src.utils.model_loader import load_qwen_model  # 添加这行
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import pandas as pd
import numpy as np
import re
from scipy.stats import spearmanr
from datetime import datetime

# ========== 配置（与b8.py一致）==========
MODEL_NAME = "Qwen/Qwen3-8B"
MAX_LENGTH = 2048

# ========== 路径 ==========
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'b8')
ABLATION_DIR = os.path.join(BASE_DIR, 'results', 'ablation')
os.makedirs(ABLATION_DIR, exist_ok=True)

# 关键修正：固定参考文件，不再随机抽
REFERENCE_PATH = os.path.join(BASE_DIR, 'data', 'final', 'reference.csv')
TEST_ORIG_PATH = os.path.join(BASE_DIR, 'data', 'test', 'test_ablation.csv')  # 原始测试集
DES_PATH = os.path.join(RESULTS_DIR, 'predictions.csv')  # B8生成的脱敏结果

print("=== B8 Ablation: Original vs Desensitized ===")
print("原则：参考文本固定（根据显存15/5/0句），仅测试文本变化")

# 前置检查
if not os.path.exists(REFERENCE_PATH):
    raise FileNotFoundError(f"参考文件不存在: {REFERENCE_PATH}")
if not os.path.exists(TEST_ORIG_PATH):
    raise FileNotFoundError(f"原始测试文件不存在: {TEST_ORIG_PATH}")
if not os.path.exists(DES_PATH):
    raise FileNotFoundError(f"脱敏结果不存在: {DES_PATH}\n请先运行 b8.py（使用脱敏test.csv）")

def load_model_and_determine_capacity():
    """加载模型（使用统一加载器，带OOM降级）"""
    strategies = [15, 5, 0]  # 尝试15句，OOM则降级
    
    for num_ex in strategies:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            print(f"尝试加载8B模型 (策略: {num_ex}句参考)...")
            # 使用统一加载器（会自动下载到 models/Qwen--Qwen3-8B/）
            model, tokenizer = load_qwen_model(MODEL_NAME, num_bits=4, auto_download=True)
            print(f"✓ 模型加载成功，使用 {num_ex} 句参考")
            return model, tokenizer, num_ex
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                print(f"OOM，清理显存并降级...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise e
    
    raise RuntimeError("所有加载策略均失败（8B模型无法加载）")

def build_prompt(reference_texts, test_text):
    """与b8.py一致"""
    if not reference_texts:
        prompt = f"""请为以下游戏角色台词文本的风格强度评分（0-10分，保留1位小数）：
文本：{test_text}
分数："""
    else:
        ref_block = "\n".join([f"{i+1}. {t}" for i, t in enumerate(reference_texts)])
        prompt = f"""以下是一个游戏角色的台词示例文本（共{len(reference_texts)}句）：
{ref_block}

请基于以上文本的风格特征，为以下测试文本的风格强度评分（0-10分，保留1位小数）：
文本：{test_text}
分数："""
    return prompt

# ========== 加载模型（带OOM降级）==========
print("加载模型...")
model, tokenizer, num_examples = load_model_and_determine_capacity()
print(f"最终配置: {num_examples}句参考文本\n")

# ========== 加载数据 ==========
# 读取固定参考文本（根据OOM降级结果取前N句）
ref_df = pd.read_csv(REFERENCE_PATH)
if len(ref_df) < num_examples and num_examples > 0:
    raise ValueError(f"reference.csv 只有 {len(ref_df)} 行，需要至少 {num_examples} 行")

if num_examples > 0:
    reference_texts = ref_df['text'].head(num_examples).tolist()
    print(f"✓ 从 reference.csv 加载 {len(reference_texts)} 句固定参考文本")
else:
    reference_texts = []
    print("✓ 纯Zero-shot模式（无参考文本）")

# 读取原始测试集（消融变量）
test_df = pd.read_csv(TEST_ORIG_PATH)
print(f"✓ 原始测试集: {len(test_df)} 句")

# ========== 推理（原始文本）==========
print("开始推理原始文本...")
scores_orig = []
raw_outputs = []

for idx, row in test_df.iterrows():
    text = row['text']
    prompt = build_prompt(reference_texts, text)
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    
    # 解析数字
    score = 50.0
    try:
        numbers = re.findall(r'\d+\.?\d*', response)
        if numbers:
            score = float(numbers[0])
            if score > 10:
                score = score / 10
            score = max(0, min(10, score)) * 10  # 0-100制
    except:
        pass
    
    scores_orig.append(score)
    raw_outputs.append(response)
    
    if (idx + 1) % 10 == 0:
        print(f"  进度: {idx + 1}/{len(test_df)}")

# 保存原始结果
df_orig = pd.DataFrame({
    'text': test_df['text'],
    'b8_score': scores_orig,
    'num_examples': [num_examples] * len(scores_orig),  # 记录实际配置
    'raw_output': raw_outputs
})
orig_output_path = os.path.join(RESULTS_DIR, 'predictions_original.csv')
df_orig.to_csv(orig_output_path, index=False, encoding='utf-8-sig')
print(f"\n✓ 原始结果保存至: {orig_output_path}")
print(f"  分数范围: [{min(scores_orig):.2f}, {max(scores_orig):.2f}]")

# ========== 对比脱敏版本 ==========
df_des = pd.read_csv(DES_PATH)
scores_des = df_des['b8_score'].values

# 检查配置一致性
des_num_examples = df_des['num_examples'].iloc[0] if 'num_examples' in df_des.columns else "未知"
if str(des_num_examples) != str(num_examples):
    print(f"\n  警告：主实验与消融实验配置不一致！")
    print(f"    主实验（脱敏）使用: {des_num_examples} 句")
    print(f"    消融实验（原始）使用: {num_examples} 句")
    print(f"    这可能会影响Spearman结果的解释")

if len(scores_des) != len(scores_orig):
    raise ValueError(f" 样本数不匹配: 原始{len(scores_orig)} vs 脱敏{len(scores_des)}")

# 计算Spearman
rho, p = spearmanr(scores_orig, scores_des)
print(f"\n一致性检验结果:")
print(f"Spearman ρ = {rho:.4f} (p={p:.4f})")
print(f"参考句数: {num_examples}（因OOM可能降级）")

# ========== 更新汇总表 ==========
summary_path = os.path.join(ABLATION_DIR, 'consistency_summary.csv')
new_record = {
    'method': f'B8_{num_examples}ex',  # 标注实际句数（如B8_5ex）
    'spearman_rho': round(rho, 4), 
    'p_value': round(p, 4)
}

try:
    if os.path.exists(summary_path):
        df_summary = pd.read_csv(summary_path)
        # 删除旧的B8记录（如果有）
        df_summary = df_summary[~df_summary['method'].str.startswith('B8')]
    else:
        df_summary = pd.DataFrame(columns=['method', 'spearman_rho', 'p_value'])
    
    df_summary = pd.concat([df_summary, pd.DataFrame([new_record])], ignore_index=True)
    df_summary = df_summary.sort_values('method').reset_index(drop=True)
    df_summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"\n已更新汇总表:")
    print(df_summary.to_string(index=False))
except Exception as e:
    print(f"无法更新汇总表: {e}")

# 诊断
print("\n诊断:")
if rho > 0.8:
    print("✓ 强一致：B8对内容脱敏鲁棒")
elif rho > 0.5:
    print("⚠ 中等一致：部分依赖词汇特征")
else:
    print("✗ 弱一致：严重依赖专有名词")

if num_examples < 15:
    print(f"\n注意：因显存限制仅使用 {num_examples} 句参考，"
          f"这可能降低风格建模能力，导致ρ偏高（词汇线索减少）")

print("=== B8输入消融 完成 ===")