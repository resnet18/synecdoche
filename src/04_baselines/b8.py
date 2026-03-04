"""
B8: Qwen3-7B (INT4) 
策略：尝试15句 → OOM则降级为5句 → 再OOM则纯Zero-shot
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import pandas as pd
import os
from pathlib import Path

# ========== 配置 ==========
MODEL_NAME = "Qwen/Qwen3-7B"
MAX_LENGTH = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'b8')
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'final', 'train.csv')
TEST_PATH = os.path.join(BASE_DIR, 'data', 'test', 'test_positive.csv')
OUTPUT_PATH = os.path.join(RESULTS_DIR, 'predictions.csv')
OOM_LOG_PATH = os.path.join(RESULTS_DIR, 'oom_log.txt')

print("=== B8: Qwen3-7B (带OOM降级预案) ===")

def load_model_with_fallback():
    """尝试加载模型，根据OOM情况返回配置"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    try:
        print("尝试加载7B模型 (15句参考)...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        print("15句模式加载成功")
        return model, tokenizer, 15  # 返回可用示例数
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            print(f"15句模式OOM: {e}")
            
            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 尝试5句
            try:
                print("降级：尝试5句参考...")
                tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True
                )
                print("5句模式加载成功")
                return model, tokenizer, 5
                
            except RuntimeError as e2:
                if "out of memory" in str(e2).lower():
                    print(f"5句也OOM，转为纯Zero-shot")
                    # 记录OOM日志
                    with open(OOM_LOG_PATH, 'w', encoding='utf-8') as f:
                        f.write(f"{MODEL_NAME} OOM\n尝试15句失败\n尝试5句失败\n时间: {pd.Timestamp.now()}")
                    return None, None, 0
                raise e2
        raise e

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

请基于以上文本的风格特征，为以下测试文本的风格强度评分（0-10分，保留1位小数）：
文本：{test_text}
分数："""
    return prompt

# ========== 主流程 ==========
model, tokenizer, num_examples = load_model_with_fallback()

if model is None:
    print("❌ 模型完全无法加载，终止")
    exit()

print(f"最终配置: {num_examples}句参考文本")

# 加载数据
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# 抽取参考文本
if num_examples > 0:
    reference_texts = train_df['text'].sample(n=num_examples, random_state=325).tolist()
else:
    reference_texts = []

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
        import re
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
        print(f"  已处理 {idx + 1}/{len(test_df)} 条")

# 保存
results = pd.DataFrame({
    'text': test_df['text'],
    'b8_score': scores,
    'num_examples': [num_examples] * len(scores),  # 记录实际用了几句
    'raw_output': raw_outputs
})
results.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')

print(f"\n结果保存至: {OUTPUT_PATH}")
print(f"使用参考句数: {num_examples}")
print(f"分数统计: min={min(scores):.2f}, max={max(scores):.2f}, mean={sum(scores)/len(scores):.2f}")
print("=== B8 完成 ===")