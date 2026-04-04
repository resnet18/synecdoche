"""
B8: Qwen3-8B (INT4) - 最终测试集推理脚本
策略：固定参考文本 → 尝试15句推理 → OOM则降级5句 → 再OOM则纯Zero-shot
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

# ---------- 显存限制代码 ----------
if os.getenv('CUDA_MEMORY_FRACTION'):
    torch.cuda.set_per_process_memory_fraction(float(os.getenv('CUDA_MEMORY_FRACTION')))
    print(f"显存限制: {float(os.getenv('CUDA_MEMORY_FRACTION'))*100:.0f}%")
# -------------------------------------

# ========== 配置 ==========
MODEL_NAME = "Qwen/Qwen3-8B"  # 8B（16G显存临界，支持OOM降级）
MAX_LENGTH = 2048
TARGET_EXAMPLES = 15  # 目标参考句数（可能因OOM降级）

# ========== 路径 ==========
# 输入：风格锚点（固定参考文本，与预实验相同）
REFERENCE_PATH = os.path.join(BASE_DIR, 'data', 'final', 'reference.csv')

# 输入：最终测试集（210句完整测试集）
TEST_PATH = os.path.join(BASE_DIR, 'data', 'test', 'test_full.csv')

# 输出：最终验证结果目录
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'final_validation', 'b8')
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(RESULTS_DIR, 'predictions.csv')

print("=== B8: Qwen3-8B - 最终测试集推理 ===")

# ========== 加载模型 ==========
print(f"加载模型: {MODEL_NAME}...")
try:
    model, tokenizer = load_qwen_model(MODEL_NAME, num_bits=4, auto_download=True)
    print("✓ 模型加载完成")
except Exception as e:
    print(f"模型加载失败: {e}")
    exit()

# ========== 加载数据 ==========
if not os.path.exists(REFERENCE_PATH):
    raise FileNotFoundError(f"参考文件不存在: {REFERENCE_PATH}")
if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(f"测试文件不存在: {TEST_PATH}")

ref_df = pd.read_csv(REFERENCE_PATH)
if len(ref_df) < 1:
    raise ValueError("reference.csv 为空")

test_df = pd.read_csv(TEST_PATH)
print(f"✓ 测试集: {len(test_df)} 句")
print(f"✓ 参考库: {len(ref_df)} 句（将取前N句，N∈{15,5,0}）")

# ========== 带OOM降级的推理函数 ==========
def run_with_examples(num_examples):
    """
    尝试用指定数量的参考句进行推理
    返回: (scores_list, outputs_list, success_bool)
    """
    if num_examples > 0:
        reference_texts = ref_df['text'].head(num_examples).tolist()
    else:
        reference_texts = []
    
    def build_prompt(test_text):
        if not reference_texts:
            return f"""请为以下游戏角色台词文本的风格强度评分（0-10分，保留1位小数）：
文本：{test_text}
分数："""
        ref_block = "\n".join([f"{i+1}. {t}" for i, t in enumerate(reference_texts)])
        return f"""以下是一个游戏角色的台词示例文本（共{len(reference_texts)}句）：
{ref_block}

请基于以上文本的风格特征，为以下测试文本的风格强度评分（0-10分，保留1位小数）：
文本：{test_text}
分数："""
    
    scores = []
    raw_outputs = []
    
    for idx, row in test_df.iterrows():
        test_text = row['text']
        prompt = build_prompt(test_text)
        
        try:
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
            
            # 解析分数（0-10分制，与项目其他基线保持一致）
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
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                # OOM，清理显存并返回失败
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return None, None, False
            raise e
    
    return scores, raw_outputs, True

# ========== 执行降级策略：15 → 5 → 0 ==========
strategies = [15, 5, 0]
final_scores = None
final_outputs = None
final_num = 0

for strategy in strategies:
    print(f"\n尝试 {strategy} 句参考模式...")
    scores, outputs, success = run_with_examples(strategy)
    
    if success:
        final_scores = scores
        final_outputs = outputs
        final_num = strategy
        print(f"✓ {strategy} 句模式成功")
        break
    else:
        print(f"{strategy} 句OOM，清理显存并降级...")
        continue

if final_scores is None:
    print("所有模式均失败（包括纯Zero-shot）")
    exit()

# ========== 保存结果（等待人类基准对齐） ==========
results = pd.DataFrame({
    'text_id': test_df.get('text_id', range(len(test_df))),
    'text': test_df['text'],
    'b8_score': final_scores,           # 0-10分制，与其他基线一致
    'num_examples': [final_num] * len(final_scores),  # 关键：记录实际使用的参考句数
    'raw_output': final_outputs
})
results.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')

print(f"\n✓ 结果已保存: {OUTPUT_PATH}")
print(f"实际使用参考句数: {final_num}（目标{TARGET_EXAMPLES}，因OOM可能降级）")
print(f"分数统计: min={min(final_scores):.2f}, max={max(final_scores):.2f}, mean={sum(final_scores)/len(final_scores):.2f}, std={pd.Series(final_scores).std():.2f}")

if final_num < TARGET_EXAMPLES:
    print(f"\n显存限制警告：已从{TARGET_EXAMPLES}句降级至{final_num}句")
    print(f"    这会削弱风格建模能力，但保证了物理可执行性")

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