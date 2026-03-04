"""
B8: Qwen3-7B (INT4) 
策略：读取固定参考文本 → 尝试15句 → OOM则降级为5句 → 再OOM则纯Zero-shot
关键修正：使用固定 reference.csv，不再随机抽样
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import pandas as pd
import os
import re
from pathlib import Path
from datetime import datetime

# ========== 配置 ==========
MODEL_NAME = "Qwen/Qwen3-7B"
MAX_LENGTH = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'b8')
os.makedirs(RESULTS_DIR, exist_ok=True)

# 关键修正：固定参考文件，不再从train.csv随机抽
REFERENCE_PATH = os.path.join(BASE_DIR, 'data', 'final', 'reference.csv')
TEST_PATH = os.path.join(BASE_DIR, 'data', 'test', 'test_positive.csv')
OUTPUT_PATH = os.path.join(RESULTS_DIR, 'predictions.csv')
OOM_LOG_PATH = os.path.join(RESULTS_DIR, 'oom_log.txt')

print("=== B8: Qwen3-7B (INT4 with OOM Fallback) ===")
print("原则：参考文本固定，仅调整阅读数量应对显存限制")

# 前置检查
if not os.path.exists(REFERENCE_PATH):
    raise FileNotFoundError(f"❌ 参考文件不存在: {REFERENCE_PATH}")
if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(f"❌ 测试文件不存在: {TEST_PATH}")

def load_model_and_determine_capacity():
    """
    尝试加载模型并根据显存情况确定可处理的参考句数
    顺序：15句 → 5句 → 0句（纯Zero-shot）
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    tokenizer = None
    model = None
    num_examples = 15  # 初始尝试
    
    # 尝试策略列表
    strategies = [15, 5, 0]
    
    for strategy in strategies:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            if strategy == 0:
                print("降级：纯Zero-shot模式（无参考文本）")
                tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True
                )
                num_examples = 0
                print("✓ Zero-shot模式加载成功")
                break
            else:
                print(f"尝试加载7B模型 ({strategy}句参考)...")
                tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True
                )
                num_examples = strategy
                print(f"✓ {strategy}句模式加载成功")
                break
                
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "cuda" in error_msg:
                print(f"⚠️  {strategy}句模式OOM，清理显存...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                if strategy == 5:
                    print("连续OOM，准备转为纯Zero-shot...")
                continue
            else:
                raise e
    
    if model is None:
        # 记录OOM日志
        with open(OOM_LOG_PATH, 'w', encoding='utf-8') as f:
            f.write(f"Model: {MODEL_NAME}\n")
            f.write(f"Status: OOM on all strategies (15/5/0)\n")
            f.write(f"Time: {datetime.now().isoformat()}\n")
        raise RuntimeError("❌ 所有加载策略均失败，包括Zero-shot")
    
    return model, tokenizer, num_examples

def build_prompt(reference_texts, test_text):
    """构造Prompt，支持0/5/15句参考"""
    if not reference_texts:
        # 纯Zero-shot
        prompt = f"""请为以下游戏角色台词文本的风格强度评分（0-10分，保留1位小数）：
文本：{test_text}
分数："""
    else:
        # 有参考文本
        ref_block = "\n".join([f"{i+1}. {t}" for i, t in enumerate(reference_texts)])
        prompt = f"""以下是一个游戏角色的台词示例文本（共{len(reference_texts)}句）：
{ref_block}

请基于以上文本的风格特征（如句式、用词、语气等），为以下测试文本的风格强度评分（0-10分，保留1位小数）：
文本：{test_text}
分数："""
    return prompt

# ========== 主流程 ==========
print("加载模型...")
model, tokenizer, num_examples = load_model_and_determine_capacity()

print(f"\n最终配置: {num_examples}句参考文本")

# 加载数据（关键修正：固定参考）
ref_df = pd.read_csv(REFERENCE_PATH)
if len(ref_df) < num_examples and num_examples > 0:
    raise ValueError(f"❌ reference.csv 只有 {len(ref_df)} 行，需要至少 {num_examples} 行")

# 取固定参考文本（前N句）
if num_examples > 0:
    reference_texts = ref_df['text'].head(num_examples).tolist()
    print(f"✓ 从 reference.csv 加载 {len(reference_texts)} 句固定参考文本")
else:
    reference_texts = []
    print("✓ 使用纯Zero-shot（无参考文本）")

# 读取测试集
test_df = pd.read_csv(TEST_PATH)
print(f"✓ 测试集: {len(test_df)} 句")

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
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    response = response.strip()
    
    # 解析数字
    score = 50.0
    try:
        numbers = re.findall(r'\d+\.?\d*', response)
        if numbers:
            score = float(numbers[0])
            if score > 10:
                score = score / 10
            score = max(0, min(10, score)) * 10  # 转为0-100制
    except:
        pass
    
    scores.append(score)
    raw_outputs.append(response)
    
    if (idx + 1) % 10 == 0:
        print(f"  进度: {idx + 1}/{len(test_df)}")

# 保存结果
results = pd.DataFrame({
    'text': test_df['text'],
    'b8_score': scores,
    'num_examples': [num_examples] * len(scores),  # 记录实际用了几句（OOM降级记录）
    'raw_output': raw_outputs
})
results.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')

print(f"\n✓ 结果保存至: {OUTPUT_PATH}")
print(f"实际使用参考句数: {num_examples}（因OOM可能降级）")
print(f"分数统计: min={min(scores):.2f}, max={max(scores):.2f}, mean={sum(scores)/len(scores):.2f}")

if num_examples < 15:
    print(f"\n⚠️  注意：因显存限制已降级至 {num_examples} 句参考（原始计划15句）")
    print(f"    如需复现，请确保其他方法使用相同数量的参考句进行对比")

print("=== B8 完成 ===")