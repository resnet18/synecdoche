"""
B7 Ablation: 原始文本 vs 脱敏文本一致性检验
关键：参考文本必须固定（reference.csv），只改变测试文本（原始 vs 脱敏）
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import pandas as pd
import numpy as np
import os
import re
from scipy.stats import spearmanr

# ========== 配置（必须与b7.py一致）==========
MODEL_NAME = "Qwen/Qwen3-4B"
NUM_EXAMPLES = 15
MAX_LENGTH = 2048

# ========== 路径 ==========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'b7')
ABLATION_DIR = os.path.join(BASE_DIR, 'results', 'ablation')
os.makedirs(ABLATION_DIR, exist_ok=True)

# 关键修正：使用固定参考文件，不再从train.csv随机抽
REFERENCE_PATH = os.path.join(BASE_DIR, 'data', 'final', 'reference.csv')  # 固定参考（原始文本）
TEST_ORIG_PATH = os.path.join(BASE_DIR, 'data', 'test', 'test_positive.csv')  # 原始测试文本
DES_PATH = os.path.join(RESULTS_DIR, 'predictions.csv')  # 脱敏结果（由b7.py生成）

print("=== B7 Ablation: Original vs Desensitized ===")
print("原则：参考文本固定，仅测试文本变化")

# ========== 文件存在性校验 ==========
if not os.path.exists(REFERENCE_PATH):
    raise FileNotFoundError(f"参考文件不存在: {REFERENCE_PATH}\n"
                           f"请先确保 reference.csv 存在（应与b7.py使用同一文件）")
if not os.path.exists(TEST_ORIG_PATH):
    raise FileNotFoundError(f"原始测试文件不存在: {TEST_ORIG_PATH}")
if not os.path.exists(DES_PATH):
    raise FileNotFoundError(f"脱敏结果不存在: {DES_PATH}\n"
                           f"请先运行 b7.py 生成脱敏版本预测（使用脱敏test.csv）")

# ========== 加载模型 ==========
print(f"加载模型: {MODEL_NAME} (INT4)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
print("模型加载完成")

# ========== 加载数据（关键修正：固定参考） ==========
# 读取固定参考文本（与b7.py完全一致）
ref_df = pd.read_csv(REFERENCE_PATH)
if len(ref_df) < NUM_EXAMPLES:
    raise ValueError(f"reference.csv 只有 {len(ref_df)} 行，需要至少 {NUM_EXAMPLES} 行")

reference_texts = ref_df['text'].head(NUM_EXAMPLES).tolist()
print(f"固定参考文本: {len(reference_texts)} 句（来自 reference.csv）")

# 读取原始测试集（消融变量）
test_df = pd.read_csv(TEST_ORIG_PATH)
print(f"原始测试集: {len(test_df)} 句")

# ========== Prompt构建（与b7.py完全一致） ==========
def build_prompt(reference_texts, test_text):
    ref_block = "\n".join([f"{i+1}. {t}" for i, t in enumerate(reference_texts)])
    prompt = f"""以下是一个游戏角色的台词示例文本（共{len(reference_texts)}句）：
{ref_block}

请基于以上文本的风格特征（如句式、用词、语气等），为以下测试文本的风格强度评分（0-10分，保留1位小数）：
文本：{test_text}
分数："""
    return prompt

# ========== 推理（原始测试文本） ==========
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
            score = max(0, min(10, score)) * 10  # 转为0-100制存储
    except:
        pass
    
    scores_orig.append(score)
    raw_outputs.append(response)
    
    if (idx + 1) % 10 == 0:
        print(f"  进度: {idx + 1}/{len(test_df)}")

# 保存原始分数
df_orig = pd.DataFrame({
    'text': test_df['text'],
    'b7_score': scores_orig,
    'raw_output': raw_outputs
})
orig_output_path = os.path.join(RESULTS_DIR, 'predictions_original.csv')
df_orig.to_csv(orig_output_path, index=False, encoding='utf-8-sig')
print(f"✓ 原始结果保存至: {orig_output_path}")
print(f"  分数范围: [{min(scores_orig):.2f}, {max(scores_orig):.2f}]")

# ========== 对比脱敏版本 ==========
df_des = pd.read_csv(DES_PATH)
scores_des = df_des['b7_score'].values

if len(scores_des) != len(scores_orig):
    raise ValueError(f"❌ 样本数不匹配: 原始{len(scores_orig)} vs 脱敏{len(scores_des)}")

# 计算Spearman（两者都是0-100制，直接对比）
rho, p = spearmanr(scores_orig, scores_des)
print(f"\n一致性检验结果:")
print(f"Spearman ρ = {rho:.4f} (p={p:.4f})")

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
    df_summary = df_summary.sort_values('method').reset_index(drop=True)
    df_summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"\n已更新汇总表:")
    print(df_summary.to_string(index=False))
except Exception as e:
    print(f"无法更新汇总表: {e}")

# 诊断
print("\n诊断:")
if rho > 0.8:
    print("强一致 (ρ>0.8): B7对内容脱敏鲁棒，主要捕捉结构特征")
elif rho > 0.5:
    print("中等一致 (0.5<ρ<0.8): 部分依赖词汇表面特征")
else:
    print("弱一致 (ρ<0.5): 严重依赖专有名词，脱敏后失效")

print("=== B7 Ablation 完成 ===")