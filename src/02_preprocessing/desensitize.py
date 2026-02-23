import pandas as pd
import jieba
import re
from collections import Counter

# ========== 1. 加载通用高频词表 ==========
def load_common_words(top_n=8000):
    jieba.initialize()
    sorted_words = sorted(jieba.dt.FREQ.items(), key=lambda x: x[1], reverse=True)
    common_set = set([w for w, f in sorted_words if len(w) >= 2][:top_n])
    print(f"已加载通用词表，共 {len(common_set)} 个词")
    return common_set

# ========== 2. 核心脱敏函数 ==========
def desensitize_text(text, common_set, hardcoded_map):
    # 2.1 硬编码层（必须包含所有明确已知的角色名/地名）
    for old, new in hardcoded_map.items():
        text = text.replace(old, new)
    
    # 2.2 英文代号（用负向环视，不依赖单词边界）
    text = re.sub(r'(?<![a-zA-Z])[a-zA-Z][a-zA-Z0-9\-]*(?![a-zA-Z])', '[代号]', text)
    
    # 2.3 模式捕获（扩大后缀列表！）
    # 去掉\b，改用中文语境匹配
    pattern = r'([\u4e00-\u9fa5]{1,3}(?:斯|兹|娅|希|卡|尔|亚|纶|娜|拉|恩|蒂|克|玛|莉))(?![\u4e00-\u9fa5])'
    candidates = re.findall(pattern, text)
    
    freq = Counter(candidates)
    
    # 构建替换映射（不在通用词表才替换）
    replacement_map = {}
    for word, count in freq.items():
        if word not in common_set and len(word) >= 2:
            replacement_map[word] = '[N]'
    
    # 按长度降序替换（长词优先）
    for word in sorted(replacement_map.keys(), key=len, reverse=True):
        text = text.replace(word, replacement_map[word])
    
    return text, replacement_map

# ========== 3. 主流程 ==========
def process_corpus(input_file, output_file, debug_mode=False):
    print("正在构建通用词表...")
    common_set = load_common_words(8000)
    
    # 关键修复：硬编码列表必须包含所有高频专有词
    hardcoded = {
        # ========== 第一层：完整称号（必须比单个词长）==========
        "血魔大君": "[头衔]",
        "食腐者之王": "[头衔]",
        "文明的存续": "[概念]",
        "提卡兹之血": "[概念]",

        # ========== 第二层：基础种族/概念（短词）==========
        # 角色名（即使结巴认识也要强制替换）
        "阿米娅": "[人名]",
        "阿斯卡纶": "[人名]", 
        "凯尔希": "[人名]",
        "逻各斯": "[人名]",
        "特蕾西娅": "[人名]",
        "特雷西斯": "[人名]",
        "伊内斯": "[人名]",
        "赫德雷": "[人名]",
        "闪灵": "[人名]",
        "明椒": "[人名]",
        "维什戴尔": "[人名]",
        "变形者": "[人名]",
        "号角": "[人名]",
        "推进之王": "[代号]",
        "杜卡雷": "[人名]",
        "孽茨雷": "[人名]",
        "萨卢斯": "[人名]",
        "可露希尔": "[人名]",
        
        # 地区/种族/组织
        "罗德岛": "[地名]",
        "巴别塔": "[组织]", 
        "王庭": "[组织]",
        "自救军": "[组织]",
        "军事委员会": "[组织]",
        "卡兹戴尔": "[地名]",
        "泰拉": "[地名]",
        "伊比利亚": "[地名]",
        "阿戈尔": "[地名]",
        "伦蒂尼姆": "[地名]",
        "布伦特伍德": "[地名]",
        "维多利亚": "[地名]",
        "雷姆必拓": "[地名]",
        "切尔诺伯格": "[地名]",
        "河谷": "[地名]",
        "血魔": "[种族]",
        "萨卡兹": "[种族]",
        "卡特斯": "[种族]",
        "萨科塔": "[种族]",
        "提卡兹": "[种族]",
        "女妖": "[种族]",
        "食腐者": "[种族]",
        "死魂灵": "[种族]",
        "阿斯兰": "[种族]",
        "红龙": "[种族]",
        "德拉克": "[种族]",

        # 概念
        "源石": "[概念]",
        "众魂": "[概念]",
        "巨兽": "[概念]",
        "矿石病": "[疾病]",
        "感染者": "[身份]",
        "法术大师": "[装置]",
        "干员": "[身份]",
        "阿喃那": "[概念]",
        "咒言": "[术语]",      # 逻各斯专属能力词
        "咒文": "[术语]",      # 同义变体
        "巫咒": "[术语]",      # 如果有的话
        "鳞": "[动物]",
        "羽兽": "[动物]",
        "驼兽": "[动物]",
        "走兽": "[动物]", 


        
        # 称谓
        "魔王": "[称谓]",
        "博士": "[称谓]",
        "殿下": "[称谓]",
        "公爵": "[称谓]",
        "赦罪师": "[称谓]",
        "使徒": "[称谓]"
    }

    # 关键修改：按长度降序排序，长词先替换
    sorted_hardcoded = dict(sorted(hardcoded.items(), key=lambda x: len(x[0]), reverse=True))
    
    print(f"读取 {input_file}...")
    df = pd.read_csv(input_file)
    
    # 自动检测文本列
    text_col = None
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['text', 'content', 'sentence', 'line', '对话', '文本']):
            text_col = col
            break
    if text_col is None:
        text_col = df.columns[0]
    
    print(f"处理文本列: {text_col}")
    
    # 处理
    processed_texts = []
    all_replacements = {}
    
    for idx, row in df.iterrows():
        text = str(row[text_col])
        new_text, rep_map = desensitize_text(text, common_set, sorted_hardcoded)
        processed_texts.append(new_text)
        
        for k, v in rep_map.items():
            all_replacements[k] = all_replacements.get(k, 0) + 1
        
        if (idx + 1) % 100 == 0:
            print(f"已处理 {idx + 1}/{len(df)} 行")
    
    # 保存结果
    result_df = pd.DataFrame({'id': df['id'],'text': processed_texts})
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n已保存干净版本到: {output_file}")
    print(f"总行数: {len(result_df)}")
    
    # ========== 调试模式：生成候选词表 ==========
    if debug_mode:
        print("\n========== 候选词表（建议加入硬编码） ==========")
        all_candidates = set()
        
        # 重新扫描原始数据（不做替换，找漏网之鱼）
        for idx, row in df.iterrows():
            text = str(row[text_col])
            # 暴力模式：只要匹配后缀就抓
            pattern = r'[\u4e00-\u9fa5]{1,3}(?:斯|兹|娅|希|卡|尔|亚|纶|娜|拉|恩|蒂|克|玛|莉)'
            matches = re.findall(pattern, text)
            all_candidates.update(matches)
        
        # 过滤：不在硬编码、不在通用词表、不是单字、不是已替换的标签
        new_names = [w for w in all_candidates 
                     if w not in hardcoded 
                     and w not in common_set 
                     and len(w) >= 2 
                     and not w.startswith('[')]
        
        for name in sorted(new_names, key=len, reverse=True):
            print(f'"{name}": "[人名]",')
    
    # 统计
    print("\n========== 替换统计（Top 15） ==========")
    top_replaced = sorted(all_replacements.items(), key=lambda x: x[1], reverse=True)[:15]
    for word, count in top_replaced:
        print(f"{word} -> [N]  ({count}次)")

# ========== 4. 执行 ==========
if __name__ == "__main__":
    input_csv = "data/processed/logos_cleaned.csv"      
    output_csv = "data/final/logos_desensitized.csv"    
    
    # 开关调试模式：True=打印候选词表，False=不打印
    process_corpus(input_csv, output_csv, debug_mode=False)
    
    print("\n========== 验证建议 ==========")
    print("检查 data/final/logos_desensitized.csv 是否只有一列'text'，且包含[N]和[代号]")