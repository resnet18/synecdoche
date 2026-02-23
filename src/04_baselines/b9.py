import pandas as pd
import numpy as np
import os
import time
import re
from openai import OpenAI

# 这个脚本目前还有点问题，需要调整，或者重新写一个
# === 路径配置 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'b7')
os.makedirs(RESULTS_DIR, exist_ok=True)

FEW_SHOT_PATH = os.path.join(BASE_DIR, 'data', 'final', 'few_shot_examples.csv')
TEST_PATH = os.path.join(BASE_DIR, 'data', 'test', 'test_positive.csv')
OUTPUT_PRED = os.path.join(RESULTS_DIR, 'predictions.csv')

# === API配置 ===
# 记得把api_keys.txt里面的占位符换成自己真实的api key
# 从项目根目录的api_keys文件读取（第一行格式：deepseek api: sk-...）
api_key_path = os.path.join(BASE_DIR, 'api_keys.txt')
with open(api_key_path, 'r', encoding='utf-8') as f:
    first_line = f.readline().strip()
    # 提取sk-开头的部分（到空格或行尾）
    if 'sk-' in first_line:
        start = first_line.index('sk-')
        end = first_line.find(' ', start)
        api_key = first_line[start:end] if end != -1 else first_line[start:]
    else:
        # 备用：取冒号后的内容
        api_key = first_line.split(':')[-1].strip()

print(f"已加载API Key: {api_key[:10]}...")  # 打印前10位确认加载成功

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

MODEL = "deepseek-reasoner"   # 使用思考模型

print("=== B7 DeepSeek API基线（开启思考模式）===")

# 读取数据
few_shot_df = pd.read_csv(FEW_SHOT_PATH)
test_df = pd.read_csv(TEST_PATH)

examples = few_shot_df['text'].tolist()
test_texts = test_df['text'].astype(str).tolist()

print(f"Few-shot示例：{len(examples)}句")
print(f"测试集：{len(test_texts)}句")

# 构造系统提示词（15句示例平铺）
system_prompt = "你是一位文本风格分析专家。请根据提供的示例台词，评估新句子的风格相似度。\n\n示例台词（风格参考）：\n"
for i, text in enumerate(examples, 1):
    system_prompt += f"{i}. {text}\n"

system_prompt += "\n评分标准（1-10分）：\n10分 = 与示例风格极其相似（典型角色风格）\n1分 = 与示例风格完全不符\n\n请只输出1-10的整数评分，格式：评分：X"

def get_score(text, max_retries=3):
    """调用DeepSeek API获取分数（开启思考模式）"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"请评估以下句子的风格相似度：\n{text}"}
                ],
                temperature=0.1,          # 低温度，减少随机性
                max_tokens=50,             # 增大token限制，确保推理过程+答案完整
                reasoning_effort="high",   # 开启高投入思考模式
                stream=False               
            )
            
            content = response.choices[0].message.content
            
            # 解析数字
            match = re.search(r'(\d+)', content)
            if match:
                score = int(match.group(1))
                score = max(1, min(10, score))  # 限制1-10
                return score
                
        except Exception as e:
            print(f"  第{attempt+1}次尝试失败：{e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    return 5  # 失败默认中位数

# 批量调用（防限流，每句间隔0.5秒）
print("\n开始调用API（开启思考模式）...")
scores = []

for i, text in enumerate(test_texts, 1):
    score = get_score(text)
    scores.append(score)
    print(f"  [{i}/{len(test_texts)}] 评分：{score}")
    time.sleep(0.5)  # 防止速率限制

# 转0-100分制并保存
scores_100 = [s * 10 for s in scores]

results = pd.DataFrame({
    'text': test_texts,
    'b7_score': scores_100,
    'b7_raw': scores
})
results.to_csv(OUTPUT_PRED, index=False, encoding='utf-8-sig')

print(f"\n结果保存至：{OUTPUT_PRED}")
print(f"分数统计：min={min(scores_100):.2f}, max={max(scores_100):.2f}, mean={np.mean(scores_100):.2f}")