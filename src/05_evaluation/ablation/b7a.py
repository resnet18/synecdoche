"""
B7 Ablation: 原始文本 vs 脱敏文本一致性检验
关键：参考文本固定（reference.csv），仅测试文本变化（原始 vs 脱敏）
"""
import os
import sys
from pathlib import Path

# 关键修正1：向上4层到项目根（因为b7a.py在src/05_evaluation/ablation/）
BASE_DIR = str(Path(__file__).resolve().parents[3])
# 或者：BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, BASE_DIR)
from src.utils.model_loader import load_qwen_model

# 关键修正2：强制离线，避免Qwen tokenizer联网查询
# os.environ["HF_HUB_OFFLINE"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import pandas as pd
import re
from scipy.stats import spearmanr

# ========== 配置 ==========
MODEL_NAME = "Qwen/Qwen3-4B"
NUM_EXAMPLES = 15
MAX_LENGTH = 2048

# ========== 路径 ==========
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'b7')
ABLATION_DIR = os.path.join(BASE_DIR, 'results', 'ablation')
os.makedirs(ABLATION_DIR, exist_ok=True)

REFERENCE_PATH = os.path.join(BASE_DIR, 'data', 'final', 'reference.csv')
TEST_ORIG_PATH = os.path.join(BASE_DIR, 'data', 'test', 'test_ablation.csv')
DES_PATH = os.path.join(RESULTS_DIR, 'predictions.csv')

print("=== B7 Ablation: Original vs Desensitized ===")
print(f"项目根目录: {BASE_DIR}")

# 文件校验
if not os.path.exists(REFERENCE_PATH):
    raise FileNotFoundError(f"参考文件不存在: {REFERENCE_PATH}")
if not os.path.exists(TEST_ORIG_PATH):
    raise FileNotFoundError(f"原始测试文件不存在: {TEST_ORIG_PATH}")
if not os.path.exists(DES_PATH):
    raise FileNotFoundError(f"脱敏结果不存在: {DES_PATH}\n请先运行 b7.py 生成脱敏版本预测")

# ========== 加载模型 ==========
print(f"加载模型: {MODEL_NAME} (INT4)...")
model, tokenizer = load_qwen_model(MODEL_NAME, num_bits=4, auto_download=True)
print("✓ 模型加载完成")

# ========== 加载数据（固定reference） ==========
ref_df = pd.read_csv(REFERENCE_PATH)
if len(ref_df) < NUM_EXAMPLES:
    raise ValueError(f"reference.csv 只有 {len(ref_df)} 行，需要至少 {NUM_EXAMPLES} 行")

reference_texts = ref_df['text'].head(NUM_EXAMPLES).tolist()
print(f"✓ 固定参考文本: {len(reference_texts)} 句")

test_df = pd.read_csv(TEST_ORIG_PATH)
print(f"✓ 原始测试集: {len(test_df)} 句")

# ========== Prompt构建 ==========
def build_prompt(reference_texts, test_text):
    ref_block = "\n".join([f"{i+1}. {t}" for i, t in enumerate(reference_texts)])
    return f"""以下是一个游戏角色的台词示例文本（共{len(reference_texts)}句）：
{ref_block}

请基于以上文本的风格特征（如句式、用词、语气等），为以下测试文本的风格强度评分（0-10分，保留1位小数）：
文本：{test_text}
分数："""

# ========== 推理（原始文本） ==========
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
    
    score = 50.0
    try:
        numbers = re.findall(r'\d+\.?\d*', response)
        if numbers:
            score = float(numbers[0])
            if score > 10:
                score = score / 10
            score = max(0, min(10, score)) * 10
    except:
        pass
    
    scores_orig.append(score)
    raw_outputs.append(response)
    
    if (idx + 1) % 10 == 0:
        print(f"  进度: {idx + 1}/{len(test_df)}")

# 保存原始结果
df_orig = pd.DataFrame({
    'text': test_df['text'],
    'b7_score': scores_orig,
    'raw_output': raw_outputs
})
orig_output_path = os.path.join(RESULTS_DIR, 'predictions_original.csv')
df_orig.to_csv(orig_output_path, index=False, encoding='utf-8-sig')
print(f"\n✓ 原始结果保存至: {orig_output_path}")

# ========== 对比脱敏版本 ==========
df_des = pd.read_csv(DES_PATH)
scores_des = df_des['b7_score'].values

if len(scores_des) != len(scores_orig):
    raise ValueError(f"样本数不匹配: 原始{len(scores_orig)} vs 脱敏{len(scores_des)}")

rho, p = spearmanr(scores_orig, scores_des)
print(f"\n一致性检验结果: Spearman ρ = {rho:.4f} (p={p:.4f})")

# ========== 更新汇总表 ==========
summary_path = os.path.join(ABLATION_DIR, 'consistency_summary.csv')
new_record = {'method': 'B7', 'spearman_rho': round(rho, 4), 'p_value': round(p, 4)}

try:
    if os.path.exists(summary_path):
        df_summary = pd.read_csv(summary_path)
        df_summary = df_summary[df_summary['method'] != 'B7']
    else:
        df_summary = pd.DataFrame(columns=['method', 'spearman_rho', 'p_value'])
    
    df_summary = pd.concat([df_summary, pd.DataFrame([new_record])], ignore_index=True)
    df_summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"\n已更新汇总表:\n{df_summary.to_string(index=False)}")
except Exception as e:
    print(f"无法更新汇总表: {e}")

# 诊断
print("\n诊断:")
if rho > 0.8:
    print("✓ 强一致 (ρ>0.8): B7对内容脱敏鲁棒")
elif rho > 0.5:
    print("中等一致 (0.5<ρ<0.8): 部分依赖词汇特征")
else:
    print("✗ 弱一致 (ρ<0.5): 严重依赖专有名词")

print("=== B7 输入消融 完成 ===")