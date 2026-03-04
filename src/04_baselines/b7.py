"""
B7: Qwen3-4B (INT4)
策略：读取固定15句参考文本（reference.csv）→ 归纳风格 → 打分
修正：移除随机抽样，强制使用 reference.csv 保证实验可复现
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import pandas as pd
import numpy as np
import os
import re
from pathlib import Path

# ========== 配置 ==========
MODEL_NAME = "Qwen/Qwen3-4B"
NUM_EXAMPLES = 15  # 读取前15句参考文本（可调整，但必须是固定顺序）
MAX_LENGTH = 2048  # 总长度限制
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========== 路径 ==========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'b7')
os.makedirs(RESULTS_DIR, exist_ok=True)

# 关键修正：使用固定 reference.csv，不再从 train.csv 随机抽
REFERENCE_PATH = os.path.join(BASE_DIR, 'data', 'final', 'reference.csv')
TEST_PATH = os.path.join(BASE_DIR, 'data', 'test', 'test_positive.csv')
OUTPUT_PATH = os.path.join(RESULTS_DIR, 'predictions.csv')

print("=== B7: Qwen3-4B (Zero-shot with FIXED reference) ===")

# ========== 验证文件存在 ==========
if not os.path.exists(REFERENCE_PATH):
    raise FileNotFoundError(f"参考文件不存在: {REFERENCE_PATH}\n"
                           f"请确保 data/final/reference.csv 存在且包含 {NUM_EXAMPLES} 句参考文本")
if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(f"测试文件不存在: {TEST_PATH}")

# ========== 加载模型（INT4） ==========
print(f"加载模型: {MODEL_NAME} (INT4)...")
print(f"设备: {DEVICE}")

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
print("✓ 模型加载完成")

# ========== 加载数据（关键修正点） ==========
print("读取数据...")
# 读取固定参考文本（不再随机！）
ref_df = pd.read_csv(REFERENCE_PATH)
if len(ref_df) < NUM_EXAMPLES:
    raise ValueError(f"reference.csv 只有 {len(ref_df)} 行，需要至少 {NUM_EXAMPLES} 行")

# 取前 NUM_EXAMPLES 句（固定顺序，确保可复现）
reference_texts = ref_df['text'].head(NUM_EXAMPLES).tolist()
print(f"从 reference.csv 加载 {len(reference_texts)} 句固定参考文本（前{NUM_EXAMPLES}句）")

# 读取测试集
test_df = pd.read_csv(TEST_PATH)
print(f"测试集: {len(test_df)} 条文本")

# ========== 构造Prompt ==========
def build_prompt(reference_texts, test_text):
    """Zero-shot with fixed context: 给固定参考，不给分数，让模型自己悟"""
    ref_block = "\n".join([f"{i+1}. {t}" for i, t in enumerate(reference_texts)])
    
    prompt = f"""以下是一个游戏角色的台词示例文本（共{len(reference_texts)}句）：
{ref_block}

请基于以上文本的风格特征（如句式、用词、语气等），为以下测试文本的风格强度评分（0-10分，保留1位小数）：
文本：{test_text}
分数："""
    return prompt

# ========== 推理 ==========
print("开始推理...")
scores = []
raw_outputs = []

for idx, row in test_df.iterrows():
    test_text = row['text']
    prompt = build_prompt(reference_texts, test_text)
    
    # 编码（注意长度截断）
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    response = response.strip()
    
    # 解析数字（robust解析）
    score = 50.0  # 默认中间值
    try:
        numbers = re.findall(r'\d+\.?\d*', response)
        if numbers:
            score = float(numbers[0])
            if score > 10:  # 如果是0-100制，转为0-10
                score = score / 10
            score = max(0, min(10, score)) * 10  # 限制在0-10，然后转0-100
    except Exception as e:
        print(f"  ⚠️  解析失败 [{idx}]: '{response}', 使用默认值50")
    
    scores.append(score)
    raw_outputs.append(response)
    
    if (idx + 1) % 10 == 0:
        print(f"  进度: {idx + 1}/{len(test_df)}")

# ========== 保存结果 ==========
results = pd.DataFrame({
    'text': test_df['text'],
    'b7_score': scores,
    'raw_output': raw_outputs
})
results.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')

print(f"\n✓ 结果保存至: {OUTPUT_PATH}")
print(f"分数统计: min={min(scores):.2f}, max={max(scores):.2f}, mean={sum(scores)/len(scores):.2f}")
print("=== B7 完成 ===")