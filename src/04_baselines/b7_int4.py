"""
B7: Qwen3-4B (INT4) - 预实验
策略：读取固定15句参考文本 → 归纳风格 → 打分
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from src.utils.model_loader import load_qwen_model
import torch
import pandas as pd
import re

# ========== 配置 ==========
# 使用 repo_id，支持自动下载到 models/Qwen--Qwen3-4B/
MODEL_NAME = "Qwen/Qwen3-4B"  # 会自动下载到项目目录
NUM_EXAMPLES = 15
MAX_LENGTH = 2048

# ========== 路径 ==========
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'b7_int4')
os.makedirs(RESULTS_DIR, exist_ok=True)

REFERENCE_PATH = os.path.join(BASE_DIR, 'data', 'final', 'reference.csv')
TEST_PATH = os.path.join(BASE_DIR, 'data', 'test', 'test_positive.csv')
OUTPUT_PATH = os.path.join(RESULTS_DIR, 'predictions.csv')

print("=== B7: Qwen3-4B (Zero-shot with FIXED reference) ===")

# ========== 加载模型 ==========
print(f"加载模型: {MODEL_NAME}...")
# auto_download=True 会自动从 ModelScope 下载到 models/Qwen--Qwen3-4B/
model, tokenizer = load_qwen_model(MODEL_NAME, num_bits=4, auto_download=True)
print("✓ 模型加载完成")

# 在 model, tokenizer = load_qwen_model(...) 之后
if hasattr(model, 'model'):
    layer = model.model.layers[0].self_attn.q_proj
    print(f"第一线性层类型: {type(layer)}")
    if 'Linear4bit' in str(type(layer)):
        print("✓ INT4 量化成功")
    else:
        print("✗ 量化失败，仍为 FP16")

# ========== 加载数据 ==========
if not os.path.exists(REFERENCE_PATH):
    raise FileNotFoundError(f"参考文件不存在: {REFERENCE_PATH}")

ref_df = pd.read_csv(REFERENCE_PATH)
if len(ref_df) < NUM_EXAMPLES:
    raise ValueError(f"reference.csv 只有 {len(ref_df)} 行，需要至少 {NUM_EXAMPLES} 行")

reference_texts = ref_df['text'].head(NUM_EXAMPLES).tolist()
print(f"✓ 从 reference.csv 加载 {len(reference_texts)} 句固定参考文本")

test_df = pd.read_csv(TEST_PATH)
print(f"✓ 测试集: {len(test_df)} 句")

# ========== Prompt构建（Qwen3无强制思考，简洁有效） ==========
def build_prompt(reference_texts, test_text):
    ref_block = "\n".join([f"{i+1}. {t}" for i, t in enumerate(reference_texts)])
    return f"""以下是一个游戏角色的台词示例文本（共{len(reference_texts)}句）：
{ref_block}

请基于以上文本的风格特征，为以下测试文本的风格强度评分（0-10分，保留1位小数）：
文本：{test_text}
分数："""

# ========== 推理 ==========
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
    
    # 解析数字（Qwen3输出干净，直接解析）
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

# ========== 保存结果 ==========
pd.DataFrame({
    'text': test_df['text'],
    'b7_score': scores,
    'raw_output': raw_outputs
}).to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')

print(f"\n✓ 结果保存至: {OUTPUT_PATH}")
print(f"统计: min={min(scores):.2f}, max={max(scores):.2f}, mean={sum(scores)/len(scores):.2f}")
print("=== B7 完成 ===")