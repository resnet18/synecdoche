"""
B7: Qwen3-4B (FP16) - 最终测试集推理脚本
策略：读取固定15句参考文本 → 归纳风格 → 为test_full.csv（210句）打分
"""

import sys
import os

# === 路径配置（4层dirname适配final_validation路径） ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, BASE_DIR)

from src.utils.model_loader import load_qwen_model
import torch
import pandas as pd
import re

# ---------- 显存限制代码（硬编码8G） ----------
import torch
if torch.cuda.is_available():
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if total_gb > 8.5:  # 只在 16G 卡上限制
        torch.cuda.set_per_process_memory_fraction(8.0 / total_gb)
        print(f"[VRAM Limit] Hard cap 8GB on {total_gb:.1f}GB device")
            # 重置峰值统计，以便准确记录
    torch.cuda.reset_peak_memory_stats()
# -------------------------------------

# ========== 配置 ==========
MODEL_NAME = "Qwen/Qwen3-4B"  # INT4量化本地模型
NUM_EXAMPLES = 15             # 固定参考文本数量
MAX_LENGTH = 2048

# ========== 路径 ==========
# 输入：风格锚点（固定参考文本，与预实验相同）
REFERENCE_PATH = os.path.join(BASE_DIR, 'data', 'final', 'reference.csv')

# 输入：最终测试集（210句完整测试集）
TEST_PATH = os.path.join(BASE_DIR, 'data', 'test', 'test_full.csv')

# 输出：最终验证结果目录
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'final_validation', 'b7_int4')
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(RESULTS_DIR, 'predictions.csv')

print("=== B7: Qwen3-4B - 最终测试集推理 ===")

# ========== 加载模型 ==========
print(f"加载模型: {MODEL_NAME}...")
model, tokenizer = load_qwen_model(MODEL_NAME, num_bits=4, auto_download=True)
print("✓ 模型加载完成")

# ========== 加载数据 ==========
if not os.path.exists(REFERENCE_PATH):
    raise FileNotFoundError(f"参考文件不存在: {REFERENCE_PATH}")

ref_df = pd.read_csv(REFERENCE_PATH)
if len(ref_df) < NUM_EXAMPLES:
    raise ValueError(f"reference.csv 只有 {len(ref_df)} 行，需要至少 {NUM_EXAMPLES} 行")

reference_texts = ref_df['text'].head(NUM_EXAMPLES).tolist()
print(f"✓ 加载 {len(reference_texts)} 句固定参考文本（风格锚点）")

test_df = pd.read_csv(TEST_PATH)
print(f"✓ 测试集: {len(test_df)} 句")

# ========== Prompt构建 ==========
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
    
    # 解析数字（0-10分范围）
    score = 5.0  # 默认中等分数
    try:
        numbers = re.findall(r'\d+\.?\d*', response)
        if numbers:
            score = float(numbers[0])
            # 如果模型输出0-100，则归一化到0-10
            if score > 10:
                score = score / 10
            # 截断到0-10范围
            score = max(0, min(10, score))
    except:
        pass
    
    scores.append(score)
    raw_outputs.append(response)
    
    if (idx + 1) % 10 == 0:
        print(f"  进度: {idx + 1}/{len(test_df)}")

# ========== 保存结果（等待人类基准对齐） ==========
results = pd.DataFrame({
    'text_id': test_df.get('text_id', range(len(test_df))),
    'text': test_df['text'],
    'b7_score': scores,        # 已为0-10分制，与其他基线一致
    'raw_output': raw_outputs   # 保留原始输出用于调试
})
results.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')

print(f"\n✓ 结果已保存: {OUTPUT_PATH}")
print(f"统计: min={min(scores):.2f}, max={max(scores):.2f}, mean={sum(scores)/len(scores):.2f}, std={pd.Series(scores).std():.2f}")
print("待人类基准打分完成后，合并计算Spearman相关性")

# ========== 显存报告（加在这里）==========
import torch
from pathlib import Path
import os

if torch.cuda.is_available():
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    # 从环境变量获取路径（父脚本传入），或默认当前目录
    peak_file = Path(os.environ.get('PEAK_FILE_PATH', '_gpu_peak_temp.txt'))
    peak_file.write_text(str(peak_mb))
    print(f"[GPU] 峰值显存: {peak_mb:.0f} MB")
else:
    # 即使没有GPU，也写入0，避免父脚本读取失败
    peak_file = Path(os.environ.get('PEAK_FILE_PATH', '_gpu_peak_temp.txt'))
    peak_file.write_text("0")