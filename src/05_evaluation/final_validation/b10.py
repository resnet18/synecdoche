import pandas as pd
import requests
import time
import re
from pathlib import Path

# ========== 根目录（自动定位 synecdoche-main） ==========
BASE_DIR = Path(__file__).resolve().parents[3]

REFERENCE_FILE = BASE_DIR / "data" / "final" / "reference.csv"
TEST_FILE = BASE_DIR / "data" / "test" / "test_full.csv"

OUTPUT_DIR = BASE_DIR / "results" / "final_validation" / "b10"  # Kimi 版，跟 b9 区分
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ========== API配置（Kimi） ==========
API_KEY = "sk-nBLw6139J8v3HzVdkANMdiWeQHKKecD6Wkxd2eBqulLAdQ96"  # 填你的 Moonshot API Key
BASE_URL = "https://api.moonshot.cn/v1/chat/completions"
MODEL = "kimi-k2.6"

# ========== 读取数据 ==========
reference = pd.read_csv(REFERENCE_FILE)
test = pd.read_csv(TEST_FILE)

# 自动识别文本列
ref_col = [c for c in reference.columns if "text" in c.lower() or "sentence" in c.lower()][0]
test_col = [c for c in test.columns if "text" in c.lower() or "sentence" in c.lower()][0]

reference_texts = reference[ref_col].astype(str).tolist()

# 保留 text_id，如果没有就用行号
if 'text_id' in test.columns:
    test_ids = test['text_id'].astype(str).tolist()
else:
    test_ids = [str(i) for i in test.index]

test_texts = test[test_col].astype(str).tolist()

# ========== 构建 system prompt ==========
system_prompt = f"以下是一个游戏角色的台词示例（共{len(reference_texts)}句）：\n"
for i, t in enumerate(reference_texts):
    system_prompt += f"{i+1}. {t}\n"

# ========== API函数 ==========
def call_api(user_text, retry=3):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                # 强制只输出数字，不要解释
                "content": f"请基于以上文本风格打分（0-10，保留1位小数）。\n文本：{user_text}\n分数："
            }
        ],
        "temperature": 1,
    }

    for i in range(retry):
        try:
            start = time.time()
            # 超时设 90 秒，ReadTimeout 单独处理
            resp = requests.post(BASE_URL, json=payload, headers=headers, timeout=90)
            latency = (time.time() - start) * 1000

            if resp.status_code == 429:
                print(f"Rate limited (429), retry {i+1}...")
                time.sleep(5 * (i + 1))
                continue
            if resp.status_code == 504:
                print(f"Gateway timeout (504), retry {i+1}...")
                time.sleep(5 * (i + 1))
                continue

            data = resp.json()

            if "choices" not in data:
                print("API返回异常：", data)
                raise Exception("Invalid response")

            content = data["choices"][0]["message"]["content"]

            # 三级fallback分数提取
            score = 5.0
            m = re.search(r"(?:分数|打分|评分)[是为:：]?\s*(\d+(?:\.\d+)?)", content)
            if m:
                score = float(m.group(1))
            else:
                m = re.search(r"(\d+(?:\.\d+)?)\s*分", content)
                if m:
                    score = float(m.group(1))
                else:
                    nums = re.findall(r"\d+(?:\.\d+)?", content)
                    for n in nums:
                        f = float(n)
                        if 0 < f < 10:
                            score = f
                            break

            usage = data.get("usage", {})
            tokens = usage.get("total_tokens", 0)

            return score, latency, tokens

        except requests.exceptions.ReadTimeout:
            print(f"Read timeout, retry {i+1}...")
            time.sleep(5 * (i + 1))
        except Exception as e:
            print(f"retry {i+1} failed:", e)
            time.sleep(2 * (i + 1))

    return 5.0, 0, 0

# ========== 主循环 ==========
results = []

for idx, (text_id, text) in enumerate(zip(test_ids, test_texts)):
    print(f"Processing {idx+1}/{len(test_texts)}")

    score, latency, tokens = call_api(text)

    results.append({
        "text_id": text_id,
        "text": text,
        "score": score,
        "latency_ms": latency,
        "tokens_used": tokens
    })

    time.sleep(0.5)  # 限流（如果 429 就调大）

# ========== 保存结果 ==========
df = pd.DataFrame(results)

out_path = OUTPUT_DIR / "predictions.csv"
df.to_csv(out_path, index=False, encoding="utf-8")

# ========== 统计 ==========
summary = f"""
Total samples: {len(df)}
Avg score: {df['score'].mean():.4f}
Max score: {df['score'].max():.4f}
Min score: {df['score'].min():.4f}

Avg latency(ms): {df['latency_ms'].mean():.2f}
Total tokens: {df['tokens_used'].sum()}
"""

with open(OUTPUT_DIR / "api_cost_summary.txt", "w", encoding="utf-8") as f:
    f.write(summary)

print(summary)
print("Done ->", out_path)