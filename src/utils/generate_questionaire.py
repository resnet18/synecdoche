import pandas as pd
from pathlib import Path

# ========== 路径配置（根据你的实际结构调整） ==========
# 脚本所在目录的父级（项目根目录）
PROJECT_ROOT = Path(r"D:\Projects\synecdoche")

# 输入：最终测试集（已打乱，150句）
INPUT_CSV = PROJECT_ROOT / "data" / "test" / "test_full.csv"

# 输出目录
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # 确保目录存在

# 输出文件
OUTPUT_TXT = OUTPUT_DIR / "questionaire_import_wenjuanxing.txt"

# ========== 重复句陷阱配置 ==========
# 这是你的15个ICL参考句中的某一句（用于一致性检验）
DUPLICATE_SENTENCE = "并非是声音弃绝了这里，只是另一种轰鸣太过庞大，以至于再没有其他任何存在可以与之匹敌。"

# ========== 读取数据 ==========
print(f"正在读取: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV, encoding='utf-8')
TEXT_COLUMN = 'text'
texts = df[TEXT_COLUMN].tolist()

# 确保正好150句
if len(texts) < 150:
    print(f"警告：只有{len(texts)}句，已用空字符串补足到150句")
    texts = texts + [""] * (150 - len(texts))
else:
    texts = texts[:150]
    print(f"已读取前150句")

# ========== 设置重复句陷阱 ==========
# 第35题（索引34）和第115题（索引114）
texts[34] = DUPLICATE_SENTENCE   # 第35行
texts[114] = DUPLICATE_SENTENCE  # 第115行
print(f"已设置重复句陷阱（第35题和第115题）：{DUPLICATE_SENTENCE[:20]}...")

# ========== 生成问卷星格式 ==========
output_lines = []

for i, text in enumerate(texts, 1):
    # 清洗文本：替换英文逗号和换行符（避免CSV格式错乱）
    text_clean = str(text).replace(',', '，').replace('"', "'").replace('\n', ' ').replace('\r', '')
    
    # 问卷星批量格式：题型,题干,选项1,选项2,...
    # 0-5分制，6个选项
    line = f'单选,{text_clean},0,1,2,3,4,5'
    output_lines.append(line)

# ========== 保存文件 ==========
with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines))

print(f"\n生成完成！")
print(f"文件位置: {OUTPUT_TXT}")
print(f"总题数: {len(output_lines)}")
print("\n使用说明：")
print("1. 打开问卷星编辑界面")
print("2. 点击'批量添加题目' -> '从文本导入'")
print("3. 复制上述txt文件的全部内容，粘贴进去")
print("4. 系统会自动识别为150道单选题")
print("\n质量控制提示：")
print(f"   - 第35题和第115题是重复句（'{DUPLICATE_SENTENCE[:15]}...'）")
print("   - 用于后期计算Spearman一致性，剔除不靠谱的标注者")