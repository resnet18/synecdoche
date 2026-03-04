"""
B8 Ablation: 原始文本 vs 脱敏文本一致性检验（带OOM降级记录）
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import pandas as pd
import numpy as np
import os
import re
from scipy.stats import spearmanr

MODEL_NAME = "Qwen/Qwen3-7B"
MAX_LENGTH = 2048

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'b8')
ABLATION_DIR = os.path.join(BASE_DIR, 'results', 'ablation')
os.makedirs(ABLATION_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'final', 'train.csv')
TEST_ORIG_PATH = os.path.join(BASE_DIR, 'data', 'test', 'test_ablation.csv')
DES_PATH = os.path.join(RESULTS_DIR, 'predictions.csv')

print("=== B8 Ablation Test (Original vs Desensitized) ===")

# ========== 加载（带OOM降级，必须与b8.py一致）==========
def load_with_fallback():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    try:
        print("尝试15句模式...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
        )
        return model, tokenizer, 15
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("15句OOM，尝试5句...")
            torch.cuda.empty_cache()
            try:
                tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
                )
                return model, tokenizer, 5
            except RuntimeError as e2:
                if "out of memory" in str(e2).lower():
                    print("5句也OOM，转为纯Zero-shot")
                    return None, None, 0
                raise
        raise

model, tokenizer, num_examples = load_with_fallback()
if model is None:
    print("模型完全无法加载")
    exit()

print(f"实际使用参考句数: {num_examples}")

# 加载数据
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_ORIG_PATH)

if num_examples > 0:
    reference_texts = train_df['text'].sample(n=num_examples, random_state=325).tolist()
else:
    reference_texts = []

# ========== Prompt ==========
def build_prompt(ref_texts, test_text):
    if not ref_texts:
        return f"""请为以下游戏角色台词文本的风格强度评分（0-10分，保留1位小数）：
文本：{test_text}
分数："""
    ref_block = "\n".join([f"{i+1}. {t}" for i, t in enumerate(ref_texts)])
    return f"""以下是一个游戏角色的台词示例文本（共{len(ref_texts)}句）：
{ref_block}

请基于以上文本的风格特征，为以下测试文本的风格强度评分（0-10分，保留1位小数）：
文本：{test_text}
分数："""

# ========== 推理 ==========
print("编码原始文本...")
scores_orig = []
raw_outputs = []

for idx, row in test_df.iterrows():
    prompt = build_prompt(reference_texts, row['text'])
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5, temperature=0.1, 
                               do_sample=False, pad_token_id=tokenizer.eos_token_id)
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    
    score = 50.0
    try:
        numbers = re.findall(r'\d+\.?\d*', response)
        if numbers:
            score = float(numbers[0])
            if score > 10: score = score / 10
            score = max(0, min(10, score)) * 10
    except:
        pass
    
    scores_orig.append(score)
    raw_outputs.append(response)
    
    if (idx + 1) % 10 == 0:
        print(f"  已处理 {idx + 1}/{len(test_df)} 条")

# 保存
df_orig = pd.DataFrame({
    'text': test_df['text'],
    'b8_score': scores_orig,
    'num_examples': [num_examples] * len(scores_orig),
    'raw_output': raw_outputs
})
df_orig.to_csv(os.path.join(RESULTS_DIR, 'predictions_original.csv'), index=False, encoding='utf-8-sig')

# ========== 对比 ==========
if not os.path.exists(DES_PATH):
    print(f"错误：找不到脱敏结果 {DES_PATH}")
    exit()

df_des = pd.read_csv(DES_PATH)
# 确保脱敏版本也是同样的num_examples（如果b8.py降级了，b8a.py也必须降级）
scores_des = df_des['b8_score'].values

rho, p = spearmanr(scores_des, scores_orig)
print(f"\nSpearman ρ = {rho:.4f} (p={p:.4f}) [参考句数: {num_examples}]")

# ========== 更新汇总表 ==========
summary_path = os.path.join(ABLATION_DIR, 'consistency_summary.csv')
new_record = {'method': 'B8', 'spearman_rho': round(rho, 4), 'p_value': round(p, 4)}

try:
    if os.path.exists(summary_path):
        df_summary = pd.read_csv(summary_path)
        df_summary = df_summary[df_summary['method'] != 'B8']
    else:
        df_summary = pd.DataFrame(columns=['method', 'spearman_rho', 'p_value'])
    
    df_summary = pd.concat([df_summary, pd.DataFrame([new_record])], ignore_index=True)
    df_summary = df_summary.sort_values('method').reset_index(drop=True)
    df_summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"\n已更新汇总表:")
    print(df_summary)
except Exception as e:
    print(f"警告：无法更新汇总表: {e}")

# 诊断
if rho > 0.7:
    print("\n强一致：B8对专有词不敏感")
elif rho > 0.3:
    print("\n中等一致：部分依赖专有词")
else:
    print("\n弱一致：严重词汇泄漏")

print("=== B8 Ablation 完成 ===")