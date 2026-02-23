import requests
import base64
import json
import re
import time
import os
from urllib.parse import quote

# 配置
WIKI_API = "https://prts.wiki/api.php"
TARGET = "Logos"
OUTPUT_FILE = "data/logos_all_dialogues.txt"
BATCH_SIZE = 2100

# 保持会话，减少被ban概率
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
})

def get_all_story_pages():
    """获取所有使用剧情模拟器模板的页面"""
    print("正在获取所有剧情页面列表...")
    all_pages = []
    continue_param = None
    
    while True:
        params = {
            'action': 'query',
            'list': 'embeddedin',
            'eititle': 'Template:剧情模拟器',
            'eilimit': 500,  # 每次最多500
            'format': 'json'
        }
        if continue_param:
            params['eicontinue'] = continue_param
            
        try:
            r = session.get(WIKI_API, params=params, timeout=30).json()
            pages = [p['title'] for p in r['query']['embeddedin'] if p.get('ns') == 0]
            all_pages.extend(pages)
            print(f"  已获取 {len(all_pages)} 个页面...")
            
            continue_param = r.get('continue', {}).get('eicontinue')
            if not continue_param:
                break
            time.sleep(0.5)
        except Exception as e:
            print(f"获取列表出错: {e}")
            break
    
    return all_pages

def decode_story_data(wikitext):
    """从wikitext中提取并解码剧情数据"""
    # 查找 {{剧情模拟器|data=...}} 或 {{剧情模拟器|data1=...|data2=...}}
    # 有些页面有多个data字段（多段剧情）
    pattern = r'\{\{剧情模拟器\|data\d*?=([A-Za-z0-9+/=]+)'
    matches = re.findall(pattern, wikitext)
    
    all_text = ""
    for b64_data in matches:
        try:
            # 尝试解码
            decoded = base64.b64decode(b64_data).decode('utf-8', errors='ignore')
            all_text += decoded + "\n"
        except:
            continue
    
    # 如果没找到base64，可能早期关卡是明文存储，直接返回wikitext
    if not all_text and '[name=' in wikitext:
        all_text = wikitext
    
     # ===== 新增这行：去掉 {{剧情导航}} 等 wiki 标记 =====
    all_text = re.sub(r'\{\{.*?\}\}', '', all_text)

    return all_text

def extract_logos_lines(text):
    """从解码后的文本中提取Logos台词"""
    lines = []
    
    # 匹配 [name="Logos"] 或 [name='Logos'] 或 [name=&quot;Logos&quot;]（HTML实体）
    # 容忍各种引号和空格，以及HTML实体
    text = text.replace('&quot;', '"').replace('&#39;', "'")
    
    pattern = (r'\[name\s*=\s*["\']?' + re.escape(TARGET) + 
           r'["\']?\]\s*([^\[\{\n]+?)(?=\[|\{|\n|\Z)')
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        # 清理：去掉[Character]动作标签，合并空白
        cleaned = re.sub(r'\[.*?\]', '', match)
        cleaned = ' '.join(cleaned.split())
         # ===== 新增这段：超长句可能是旁白，只保留第一句 =====
        if len(cleaned) > 120:
            first_sentence = cleaned.split('。')[0]
            if len(first_sentence) > 10:
                cleaned = first_sentence + '。'
        # 不过滤短句（保留二字咒语），但过滤纯空
        if cleaned:
            lines.append(cleaned)
    
    return lines

def process_page(title):
    """处理单个页面"""
    try:
        # 使用 action=parse 获取 wikitext（绕过网页反爬）
        params = {
            'action': 'parse',
            'page': title,
            'prop': 'wikitext',
            'format': 'json'
        }
        
        r = session.get(WIKI_API, params=params, timeout=10)
        data = r.json()
        
        if 'error' in data:
            return []
        
        wikitext = data['parse']['wikitext']['*']
        
        # 检查是否包含 Logos（快速过滤，避免解码所有页面）
        if TARGET not in wikitext and TARGET not in wikitext.replace('&quot;', '"'):
            return []
        
        # 解码剧情数据
        story_text = decode_story_data(wikitext)
        
        # 提取台词
        return extract_logos_lines(story_text)
        
    except Exception as e:
        return []

# 主程序
def main():
    # 获取所有页面
    pages = get_all_story_pages()
    total = len(pages)
    print(f"共找到 {total} 个剧情页面，开始筛选包含 {TARGET} 的...")
    
    # 断点续传：检查已有进度
    processed = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
            processed = content.count('===') // 2  # 估算已处理页面数
        print(f"检测到已有进度，跳过前 {processed} 个页面")
        pages = pages[processed:]
    
    results = []
    
    # 测试模式：只跑前BATCH_SIZE个
    test_pages = pages[:BATCH_SIZE] if BATCH_SIZE else pages
    
    for idx, page in enumerate(test_pages):
        print(f"[{idx+1}/{len(test_pages)}] 检查: {page[:40]}...", end=' ')
        
        lines = process_page(page)
        
        if lines:
            print(f"找到 {len(lines)} 句")
            results.extend([(page, line) for line in lines])
            # 实时保存，防丢失
            with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                f.write(f"\n=== {page} ===\n")
                for line in lines:
                    f.write(f"{line}\n")
        else:
            print("无台词")
        
        # 礼貌延迟，防止API限流
        time.sleep(0.3)
    
    print(f"\n完成！本次共找到 {len(results)} 句台词")
    print(f"结果已追加保存至: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()