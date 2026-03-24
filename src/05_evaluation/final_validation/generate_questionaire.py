import pandas as pd
from pathlib import Path

# ========== 路径配置 ==========
PROJECT_ROOT = Path(r"D:\Projects\synecdoche")
INPUT_CSV = PROJECT_ROOT / "data" / "test" / "test_full.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_TXT = OUTPUT_DIR / "questionaire_import_wenjuanxing.txt"

DUPLICATE_SENTENCE = "并非是声音弃绝了这里，只是另一种轰鸣太过庞大，以至于再没有其他任何存在可以与之匹敌。"

# ========== 读取数据 ==========
df = pd.read_csv(INPUT_CSV, encoding='utf-8')
texts = df['text'].tolist()
original_total = len(texts)
print(f"原测试集: {original_total}句")

# ========== 保存被替换的句子 ==========
# 先保存原35句和115句的内容（即将被替换）
saved_originals = []
if original_total > 34:
    saved_originals.append(texts[34])  # 原第35句
    print(f"保存原35句: {texts[34][:20]}...")
if original_total > 114:
    saved_originals.append(texts[114])  # 原第115句
    print(f"保存原115句: {texts[114][:20]}...")

# ========== 设置陷阱（覆盖原位置） ==========
if original_total > 34:
    texts[34] = DUPLICATE_SENTENCE  # 第35题陷阱
if original_total > 114:
    texts[114] = DUPLICATE_SENTENCE  # 第115题陷阱

# ========== 追加被保存的原句 + 第三次陷阱 ==========
texts.extend(saved_originals)  # 把被踢出去的2句加回来（变成第211、212题）
texts.append(DUPLICATE_SENTENCE)  # 第三次陷阱（第213题）

final_total = len(texts)
print(f"生成总题数: {final_total}句（原{original_total}句 + {len(saved_originals)}句挪位 + 3句陷阱）")

# ========== 生成问卷星格式 ==========
output_lines = []
for i, text in enumerate(texts, 1):
    text_clean = str(text).replace('\n', ' ').replace('\r', ' ').replace(',', '，').strip()
    output_lines.append(f"{i}.{text_clean}")
    output_lines.append("")

with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines))

print(f"\n已保存至: {OUTPUT_TXT}")
print(f"\n陷阱位置: 第35题、第115题、第{final_total}题（末尾）")
print(f"被挪位的原句位置: 第{original_total+1}题、第{original_total+2}题（也在末尾，紧接原测试集）")
print(f"\n注意：虽然原35/115句被移到了末尾，但题号变了，后续分析时需映射回原始ID")