import pandas as pd
import requests
import time
import re
from pathlib import Path
from scipy.stats import spearmanr

# ========== 根目录（自动定位 synecdoche-main） ==========
BASE_DIR = Path(__file__).resolve().parents[3]

REFERENCE_FILE = BASE_DIR / "data" / "final" / "reference.csv"
TEST_FILE = BASE_DIR / "data" / "test" / "test_ablation.csv"

OUTPUT_DIR = BASE_DIR / "results" / "b10"  # Kimi 版，跟 b9 区分
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

            score = score * 10

            usage = data.get("usage", {})
            tokens = usage.get("total_tokens", 0)

            return score, latency, tokens

        except requests.exceptions.ReadTimeout:
            print(f"Read timeout, retry {i+1}...")
            time.sleep(5 * (i + 1))
        except Exception as e:
            print(f"retry {i+1} failed:", e)
            time.sleep(2 * (i + 1))

    return 50.0, 0, 0

# ========== 主循环 ==========
results = []

for idx, text in enumerate(test_texts):
    print(f"Processing {idx+1}/{len(test_texts)}")

    score, latency, tokens = call_api(text)

    results.append({
        "text": text,
        "score": score,
        "latency_ms": latency,
        "tokens_used": tokens
    })

    time.sleep(0.5)  # 限流（如果 429 就调大）

# ========== 先组装 DataFrame ==========
df = pd.DataFrame(results)

# ========== 保存原文版结果 ==========
out_path = OUTPUT_DIR / "predictions_original.csv"
df.to_csv(out_path, index=False, encoding="utf-8")
print("Saved ->", out_path)

# ========== 消融一致性检验（原文 vs 脱敏） ==========
des_path = OUTPUT_DIR / "predictions.csv"  # 脱敏版（需提前用同脚本跑完脱敏数据）

if des_path.exists():
    df_des = pd.read_csv(des_path)
    des_scores = df_des["score"].values
    orig_scores = df["score"].values
    
    if len(des_scores) == len(orig_scores):
        rho, p = spearmanr(des_scores, orig_scores)
        
        print(f"\n【原文-脱敏一致性】Spearman ρ = {rho:.4f} (p={p:.4f})")
        
        # 汇总表路径：results/ablation/
        ablation_dir = BASE_DIR / "results" / "ablation"
        ablation_dir.mkdir(parents=True, exist_ok=True)
        summary_csv = ablation_dir / "consistency_summary.csv"
        
        record = {
            "method": "B10",  # Kimi 版标记为 B10
            "spearman_rho": round(rho, 4),
            "p_value": round(p, 4),
        }
        
        if summary_csv.exists():
            df_summary = pd.read_csv(summary_csv)
            df_summary = df_summary[df_summary["method"] != "B10"]
        else:
            df_summary = pd.DataFrame(columns=list(record.keys()))
        
        df_summary = pd.concat([df_summary, pd.DataFrame([record])], ignore_index=True)
        df_summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
        print(f"已更新汇总表: {summary_csv}")
        print(df_summary.to_string(index=False))
        
        # 诊断
        if rho > 0.7:
            print("→ 强一致：API对专有词不敏感（风格解耦良好）")
        elif rho > 0.3:
            print("→ 中等一致：部分依赖专有词（存在词汇泄漏）")
        else:
            print("→ 弱一致：严重词汇泄漏或风格漂移")
    else:
        print(f"\n警告：样本数不匹配（原文{len(orig_scores)} vs 脱敏{len(des_scores)}），跳过一致性检验")
else:
    print(f"\n提示：未找到脱敏结果 {des_path}，跳过一致性检验")

# ========== 统计摘要 ==========
summary = f"""
Total samples: {len(df)}
Avg score: {df['score'].mean():.4f}
Max score: {df['score'].max():.4f}
Min score: {df['score'].min():.4f}

Avg latency(ms): {df['latency_ms'].mean():.2f}
Total tokens: {df['tokens_used'].sum()}
"""

with open(OUTPUT_DIR / "api_cost_summary_ablation.txt", "w", encoding="utf-8") as f:
    f.write(summary)

print(summary)
print("Done")