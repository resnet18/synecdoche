import pandas as pd
from pathlib import Path

# ========== 路径配置 ==========
PROJECT_ROOT = Path(r"D:\Projects\synecdoche")
INPUT_CSV = PROJECT_ROOT / "data" / "test" / "test_full.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_TXT = OUTPUT_DIR / "questionaire_import_wenjuanxing.txt"

# ========== 重复句陷阱 ==========
DUPLICATE_SENTENCE = "并非是声音弃绝了这里，只是另一种轰鸣太过庞大，以至于再没有其他任何存在可以与之匹敌。"

# ========== 读取数据 ==========
print(f"正在读取: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV, encoding='utf-8')

TEXT_COLUMN = 'text'  # 根据实际列名修改
texts = df[TEXT_COLUMN].tolist()

# 确保150句
if len(texts) < 150:
    texts = texts + [""] * (150 - len(texts))
else:
    texts = texts[:150]

# ========== 设置重复句陷阱 ==========
texts[34] = DUPLICATE_SENTENCE   # 第35题
texts[114] = DUPLICATE_SENTENCE  # 第115题
print(f"已设置重复句陷阱（第35题和第115题）")

# ========== 生成问卷星格式（序号+文本题版） ==========
output_lines = []

for i, text in enumerate(texts, 1):
    # 清洗：去除换行符（题干中间不得换行），替换英文逗号
    text_clean = str(text).replace('\n', ' ').replace('\r', ' ').replace(',', '，').strip()
    
    # 格式：序号.题干（如：1.视死如归...）
    # 注意：序号后没有空格，直接加点和题干（根据你的示例"2.题目2"判断）
    output_lines.append(f"{i}.{text_clean}")
    
    # 空一行（表示无选项的文本题，同时分隔下一题）
    output_lines.append("")

# 保存
with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines))

print(f"\n生成完成: {OUTPUT_TXT}")
print(f"总题数: 150道单行文本题")
print("\n格式示例：")
print("1.视死如归不过是战士的终局亦是荣耀的起点")
print("")
print("2.家人们谁懂啊真是纯纯乌鱼子")
print("")
print("3.命运之线已交织无人能逃脱因果轮回")
print("")
print("...")

print(f"\n关键提示：")
print(f"- 第35题和第115题内容相同（'{DUPLICATE_SENTENCE[:15]}...'），用于计算标注者一致性")
print("- 导入后需在'问卷设置'中设置每题'必填'")
print("- 建议添加卷首语：'请输入0-5的整数（0=完全不像，5=典型风格）'")
print("- 数据导出后需清洗：剔除非0-5的输入（如'3分'、'五'、'6'、空值等）")