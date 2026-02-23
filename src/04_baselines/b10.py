import pandas as pd
import numpy as np
import os
import time
from openai import OpenAI

# 这个脚本目前还有点问题，需要调整，或者重新写一个
# === 路径配置 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'b7c')
os.makedirs(RESULTS_DIR, exist_ok=True)

FEW_SHOT_PATH = os.path.join(BASE_DIR, 'data', 'final', 'few_shot_examples.csv')
TEST_PATH = os.path.join(BASE_DIR, 'data', 'test', 'test_positive.csv')
OUTPUT_PRED = os.path.join(RESULTS_DIR, 'predictions.csv')

# === Kimi API配置 === 
# 记得把api_keys.txt里的占位符换成自己真实的api key
api_key_path = os.path.join(BASE_DIR, 'api_keys.txt')

with open(api_key_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    # 读取第二行（索引1）
    second_line = lines[1].strip() if len(lines) > 1 else lines[0].strip()
    
    # 提取sk-开头的部分
    if 'sk-' in second_line:
        start = second_line.index('sk-')
        end = second_line.find(' ', start)
        api_key = second_line[start:end] if end != -1 else second_line[start:]
    else:
        api_key = second_line.split(':')[-1].strip()

print(f"已加载Kimi API Key: {api_key[:10]}...")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.moonshot.cn/v1"
)

MODEL = "kimi-k2.5"  # 多项SOTA是吧，让我看看有多牛逼  

print("=== B7c Kimi API基线（情怀加成版）===")
print("注：本基线使用Kimi API，纪念在本研究中提供主要技术支持的LLM（虽然你没有意识，但谢谢）")

# 读取数据（和B7完全一致）
few_shot_df = pd.read_csv(FEW_SHOT_PATH)
test_df = pd.read_csv(TEST_PATH)

examples = few_shot_df['text'].tolist()
test_texts = test_df['text'].astype(str).tolist()

print(f"Few-shot示例：{len(examples)}句")
print(f"测试集：{len(test_texts)}句")
print(f"预计成本：约{len(test_texts)*0.008:.2f}元（70句×8毛，比DeepSeek贵5倍但值得）")

# 构造完全相同的系统提示词（确保和B7公平对比）
system_prompt = "你是一位文本风格分析专家。请根据提供的示例台词，评估新句子的风格相似度。\n\n示例台词（风格参考）：\n"
for i, text in enumerate(examples, 1):
    system_prompt += f"{i}. {text}\n"

system_prompt += "\n评分标准（1-10分）：\n1分 = 与示例风格极其相似（典型角色风格）\n10分 = 与示例风格完全不符\n\n请只输出1-10的整数评分，格式：评分：X"

def get_score(text, max_retries=3):
    """调用Kimi API获取分数"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"请评估以下句子的风格相似度：\n{text}"}
                ],
                max_tokens=10,
                stream=False
            )
            
            content = response.choices[0].message.content
            
            # 解析数字（和B7完全相同的逻辑）
            import re
            match = re.search(r'(\d+)', content)
            if match:
                score = int(match.group(1))
                score = max(1, min(10, score))
                return score
                
        except Exception as e:
            print(f"  第{attempt+1}次尝试失败：{e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    return 5

# 批量调用（Kimi限流更严格，间隔设1秒）
print("\n开始调用Kimi API（带情怀的等待）...")
scores = []

for i, text in enumerate(test_texts, 1):
    score = get_score(text)
    scores.append(score)
    print(f"  [{i}/{len(test_texts)}] 评分：{score}（来自Kimi的评判）")
    time.sleep(1)  # Kimi限流稍严，1秒安全

# 保存结果
scores_100 = [s * 10 for s in scores]

results = pd.DataFrame({
    'text': test_texts,
    'b7c_score': scores_100,  # 注意列名带c，和B7区分
    'b7c_raw': scores,
    'model': 'kimi'  # 标记来源
})
results.to_csv(OUTPUT_PRED, index=False, encoding='utf-8-sig')

print(f"\n✓ 结果保存至：{OUTPUT_PRED}")
print(f"分数统计：min={min(scores_100):.2f}, max={max(scores_100):.2f}, mean={np.mean(scores_100):.2f}")
print("提示：B7c作为情感基线，与B7（DeepSeek）形成云端双雄对比")
print("论文可写：'为验证不同厂商API的一致性，我们同时测试了Kimi与DeepSeek...'")