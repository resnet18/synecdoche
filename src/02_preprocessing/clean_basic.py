import pandas as pd
import re

# ========== 第1部分：处理剧情台词（从原始txt提取） ==========
input_file = "data/logos_all_dialogues.txt"

with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

cleaned_data = []
current_source = ""

for line in lines:
    line = line.strip()
    if not line:
        continue
    
    # 解析章节标题
    if line.startswith('===') and line.endswith('==='):
        current_source = line.replace('===', '').strip()
        continue
    
    # 基础清洗（删除遗漏的wiki标识）
    cleaned = re.sub(r'\[\[.*?\]\]', '', line)      
    cleaned = re.sub(r'\{\{.*?\}\}', '', cleaned)  
    cleaned = re.sub(r'\[.*?\]', '', cleaned)      
    cleaned = cleaned.strip()
    
    # 过滤（删除纯标点）
    if re.match(r'^[^\w\u4e00-\u9fff]+$', cleaned) or len(cleaned) < 3:
        continue
    
    cleaned_data.append({
        'text': cleaned,
        'source': current_source
    })

# 去重，暂时不加id
dialogue_df = pd.DataFrame(cleaned_data)
dialogue_df = dialogue_df.drop_duplicates(subset=['text']).reset_index(drop=True)
print(f"剧情台词：{len(dialogue_df)} 句（去重后）")

# ========== 第2部分：语音文本 ==========
try:
    with open("data/logos_voice.txt", "r", encoding="utf-8") as f:
        voice_lines = [l.strip() for l in f if l.strip()]
    
    voice_df = pd.DataFrame({
        "text": voice_lines,
        "source": "voice"
    })
    print(f"语音文本：{len(voice_df)} 句")
except FileNotFoundError:
    print("警告：未找到 logos_voice.txt，跳过语音部分")
    voice_df = pd.DataFrame(columns=['text', 'source'])

# ========== 第3部分：手动追加的遗漏台词 ==========
missing_lines = [
    {"text": "静止。", "source": "11-5 等价交换/END"},
    {"text": "无需道谢。", "source": "11-5 等价交换/END"},
    {"text": "我已在伦蒂尼姆见到太多无意义的屠戮。失去理智的萨卡兹理应由同胞之手阻止。", "source": "11-5 等价交换/END"},
    {"text": "所有人都总是在幻想，如果一切没有发生会怎样", "source": "11-6 演绎文明/BEG"},
    {"text": "推演没有意义。", "source": "11-6 演绎文明/BEG"},
    {"text": "胜利的号角声已在城墙上盘旋了三天三夜，可我的心为何还是如此焦灼？", "source": "11-6 演绎文明/BEG"},
    {"text": "英勇？无畏？也许是的。", "source": "11-6 演绎文明/BEG"},
    {"text": "一次激进的号召，将我们的信念凝成一团。代价，却是近在眼前的毁灭和仇恨。", "source": "11-6 演绎文明/BEG"},
    {"text": "我追随的并非君主，而是一种正直，一股勇气。", "source": "11-6 演绎文明/BEG"},
    {"text": "活着看到这个国家被暴君带着走向毁灭，远比个体毁灭更可怕。", "source": "11-6 演绎文明/BEG"},
    {"text": "我能预见她的结局，正如我知晓她的来路。", "source": "11-6 演绎文明/BEG"},
    {"text": "谁能令太阳永不下落？生出这种欲望本就是最大的贪婪。", "source": "11-6 演绎文明/BEG"},
    {"text": "为了维系这盛大的光辉，我见过无数疯狂之举。他们或是化身强盗，或是相互争斗，最终只会加速这个国家的腐朽。", "source": "11-6 演绎文明/BEG"},
    {"text": "他们不愿承认，下坠之人想要攀住日光，握在手中的却只可能是闪电。", "source": "11-6 演绎文明/BEG"},
    {"text": "人们也可以选择不再回望山巅。", "source": "11-6 演绎文明/BEG"},
    {"text": "假使他们掉转自己的视线，他们会发现面前的深渊不是深渊，而是包含无穷可能性的沃土。", "source": "11-6 演绎文明/BEG"},
    {"text": "深渊就在身前，我等可用肉身填于深渊，用鲜血烧尽残垣，给后人留下一片广阔原野。", "source": "11-6 演绎文明/BEG"},
]

missing_df = pd.DataFrame(missing_lines)
print(f"手动补充：{len(missing_df)} 句")

# ========== 第4部分：合并并统一生成id ==========
# 先合并所有数据（都不带id）
full_df = pd.concat([dialogue_df, voice_df, missing_df], ignore_index=True)

# 统一生成连续id（0, 1, 2...）
full_df['id'] = range(1, len(full_df) + 1)

# 调整列顺序
full_df = full_df[['id', 'text', 'source']]

# 保存（全新文件，不追加）
output_file = "data/logos_cleaned.csv"
full_df.to_csv(output_file, index=False, encoding='utf-8')

print(f"\n✅ 合并完成：总计 {len(full_df)} 句（id 0-{len(full_df)-1}）")
print(f"已保存至：{output_file}")
print("\n前3行示例：")
print(full_df.head(3))
print("\n最后3行示例：")
print(full_df.tail(3))