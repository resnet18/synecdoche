import re

input_file = "data/logos_all_dialogues.txt"
output_file = "data/logos_chapters_list.txt"

with open(input_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 提取所有 === 章节标题 ===
chapters = re.findall(r'=== (.+?) ===', content)

# 去重并保持顺序
seen = set()
unique_chapters = []
for ch in chapters:
    if ch not in seen:
        seen.add(ch)
        unique_chapters.append(ch)

print(f"找到 {len(unique_chapters)} 个包含Logos台词的章节：")
for i, ch in enumerate(unique_chapters, 1):
    print(f"{i}. {ch}")

# 保存为Python列表格式，明天直接复制进代码
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("# Logos出场章节列表（从第一遍爬取结果提取）\n")
    f.write("PRIORITY_PAGES = [\n")
    for ch in unique_chapters:
        # 处理引号和转义
        safe_ch = ch.replace('"', '\\"')
        f.write(f'    "{safe_ch}",\n')
    f.write("]\n")

print(f"\n已保存至 {output_file}")