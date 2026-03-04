"""
B7: Qwen3-4B (INT4)
策略：读取固定15句参考文本（reference.csv）→ 归纳风格 → 打分
修正：移除随机抽样，强制使用 reference.csv 保证实验可复现
"""
import sys
import os
# 关键：强制使用国内镜像（hf-mirror.com），不走官方 Hugging Face
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"  # 确保能联网下载（下载完后再改成1）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)  # 确保能找到utils

from src.utils.model_loader import load_qwen_model  # 使用封装好的加载器
from transformers import BitsAndBytesConfig  # 如果还需要其他配置
import torch
import pandas as pd
import numpy as np
import re

# ========== 配置 ==========
MODEL_NAME = "Qwen/Qwen3-4B"
NUM_EXAMPLES = 15
MAX_LENGTH = 2048

# ========== 路径 ==========
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'b7')
os.makedirs(RESULTS_DIR, exist_ok=True)

REFERENCE_PATH = os.path.join(BASE_DIR, 'data', 'final', 'reference.csv')
TEST_PATH = os.path.join(BASE_DIR, 'data', 'test', 'test_positive.csv')
OUTPUT_PATH = os.path.join(RESULTS_DIR, 'predictions.csv')

print("=== B7: Qwen3-4B (Zero-shot with FIXED reference) ===")

# ========== 加载模型（自动处理在线/离线） ==========
print(f"加载模型: {MODEL_NAME}...")
model, tokenizer = load_qwen_model(MODEL_NAME, num_bits=4)
print("✓ 模型加载完成")

# ========== 后续代码和你之前的一致 ==========
# 文件检查
if not os.path.exists(REFERENCE_PATH):
    raise FileNotFoundError(f"❌ 参考文件不存在: {REFERENCE_PATH}")

# 加载数据（固定reference）
ref_df = pd.read_csv(REFERENCE_PATH)
if len(ref_df) < NUM_EXAMPLES:
    raise ValueError(f"❌ reference.csv 只有 {len(ref_df)} 行，需要至少 {NUM_EXAMPLES} 行")

reference_texts = ref_df['text'].head(NUM_EXAMPLES).tolist()
print(f"✓ 从 reference.csv 加载 {len(reference_texts)} 句固定参考文本")

test_df = pd.read_csv(TEST_PATH)
print(f"✓ 测试集: {len(test_df)} 句")

# Prompt构建
def build_prompt(reference_texts, test_text):
    ref_block = "\n".join([f"{i+1}. {t}" for i, t in enumerate(reference_texts)])
    return f"""以下是一个游戏角色的台词示例文本（共{len(reference_texts)}句）：
{ref_block}

请基于以上文本的风格特征，为以下测试文本的风格强度评分（0-10分，保留1位小数）：
文本：{test_text}
分数："""

# 推理
print("开始推理...")
scores = []
raw_outputs = []

for idx, row in test_df.iterrows():
    test_text = row['text']
    prompt = build_prompt(reference_texts, test_text)
    
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
            score = max(0, min(10, score)) * 10
    except:
        pass
    
    scores.append(score)
    raw_outputs.append(response)
    
    if (idx + 1) % 10 == 0:
        print(f"  进度: {idx + 1}/{len(test_df)}")

# 保存
pd.DataFrame({
    'text': test_df['text'],
    'b7_score': scores,
    'raw_output': raw_outputs
}).to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')

print(f"\n✓ 结果保存至: {OUTPUT_PATH}")
print(f"统计: min={min(scores):.2f}, max={max(scores):.2f}, mean={sum(scores)/len(scores):.2f}")
print("=== B7 完成 ===")