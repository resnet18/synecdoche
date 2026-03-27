"""
O1-v2: 结构指纹基线（控制变量版）
目标：仅分析非标签文本的结构特征，标签内容中性化处理
排除：任何依赖标签内语义（源石vs概念）的特征
保留：标点结构、功能词密度、句法模板、占位符结构
"""

import pandas as pd
import numpy as np
import re
import os
import jieba
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# ========== 1. 纯功能词表（与内容无关，仅反映语域注册） ==========

# 文言/正式语域功能词（之乎者也，无论前后接什么内容都成立）
FUNCTION_PARTICLES = {
    # 结构助词（虚词，无实义）
    '之', '的', '得', '地', '而', '以', '于', '为', '与', '乃', '所', '被', '把',
    # 连词（逻辑关系标记）
    '并', '且', '而', '却', '但', '然', '则', '故', '因', '虽', '若', '苟',
    # 副词（语气/程度，无实指）
    '已', '亦', '皆', '尽', '俱', '均', '咸', '悉', '尽皆', '全都', '亦都',
    '尚', '且', '仍', '还', '依然', '依旧', '仍然', '仍旧', '还是',
    '必', '定', '须', '应', '当', '该', '须得', '定要', '必将', '定将',
    '勿', '毋', '莫', '不', '没', '未', '无', '非', '勿要', '不要', '不必',
    '甚', '颇', '极', '甚而', '甚至', '极其', '甚为', '很是', '相当', '非常',
    # 代词（指代功能，无实指内容）
    '其', '此', '这', '那', '斯', '兹', '之', '彼', '其', '如此', '这般',
    # 疑问/反诘（语气功能）
    '何', '胡', '焉', '安', '孰', '谁', '何妨', '何不', '何必', '何以', '为何',
    # 判断/限定（语气功能，非内容词）
    '即', '便', '就', '乃', '系', '是', '非', '即系', '即是', '便是', '就是',
    '仅', '只', '但', '唯', '惟', '只是', '仅仅', '只不过', '唯只', '惟独'
}

# 连接词（句法结构标记，与内容无关）
CLAUSE_MARKERS = {
    # 因果（逻辑框架）
    '因此', '所以', '因而', '从而', '于是', '因为', '由于', '缘于', '鉴于',
    # 转折（辩证框架）
    '但是', '然而', '不过', '只是', '却', '但', '可', '固然', '尽管', '虽然',
    # 条件（假设框架）
    '如果', '若', '若是', '假使', '假如', '倘若', '倘使', '只要', '除非', '无论',
    # 递进（层次框架）
    '而且', '并且', '况且', '何况', '甚至', '甚而', '乃至', '不仅', '不只', '不但'
}

# 语气/情态标记（正式语域指示器）
MODAL_MARKERS = {
    '无妨', '不妨', '可否', '能否', '是否', '能否', '可否',
    '恐怕', '怕是', '唯恐', '生怕', '惟恐', '大概', '大约', '约莫', '似乎', '好像',
    '应当', '应该', '须得', '必须', '定当', '理当', '该当', '自当', '本当',
    '未必', '不一定', '不见得', '难说', '未知', '未可知', '未可', '难料',
    '终究', '终将', '毕竟', '到底', '究竟', '终归', '总归', '迟早', '早晚',
    '已然', '已经', '早已', '业已', '都已', '皆已', '俱已', '全都', '全都已'
}

# 合并所有功能词（与内容无关，仅反映语域）
FUNCTION_WORDS = FUNCTION_PARTICLES | CLAUSE_MARKERS | MODAL_MARKERS

# ========== 2. 路径配置 ==========

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'o1')
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'final', 'train.csv')
VALID_PATH = os.path.join(BASE_DIR, 'data', 'final', 'validation.csv')
TEST_PATH = os.path.join(BASE_DIR, 'data', 'test', 'test_positive.csv')

OUTPUT_PRED = os.path.join(RESULTS_DIR, 'predictions.csv')
OUTPUT_CENTER = os.path.join(RESULTS_DIR, 'center_vector.npy')

print("=== O1-v2 控制变量版（结构指纹，标签内容中性化）===")

# ========== 3. 核心预处理：标签中性化 ==========

def neutralize_placeholders(text):
    """
    控制变量关键步骤：
    1. 提取标签结构信息（数量、长度、位置）
    2. 将[xxx]替换为统一标记[ENT]，消除内容语义影响
    3. 保留非标签文本用于功能词分析
    """
    if not isinstance(text, str):
        return "", [], 0
    
    # 提取标签信息
    placeholders = re.findall(r'\[([^\]]+)\]', text)
    placeholder_count = len(placeholders)
    placeholder_total_len = sum(len(p) for p in placeholders)
    
    # 标签位置模式（开头/中间/结尾）
    placeholder_positions = []
    temp_text = text
    for match in re.finditer(r'\[[^\]]+\]', text):
        start = match.start()
        placeholder_positions.append(start / len(text) if len(text) > 0 else 0)
    
    # 中性化替换：无论"源石"还是"概念"，统一为[ENT]
    neutral_text = re.sub(r'\[[^\]]+\]', '[ENT]', text)
    
    return neutral_text, placeholders, placeholder_count

def extract_structural_features(original_text):
    if not isinstance(original_text, str) or not original_text:
        return np.zeros(8)  # 改为8维
    
    # 标签中性化
    neutral_text, placeholders, ph_count = neutralize_placeholders(original_text)
    char_count = max(len(neutral_text), 1)
    
    # 分词（排除[ENT]）
    words = list(jieba.cut(neutral_text))
    non_ent_words = [w for w in words if w != '[ENT]']
    non_ent_count = max(len(non_ent_words), 1)
    
    # 标点指纹（基于中性化文本，内容无关）
    ellipsis_count = len(re.findall(r'…+|\.{3,}', neutral_text))
    ellipsis_density = ellipsis_count / char_count
    
    period_count = neutral_text.count('。') + neutral_text.count('.')
    period_density = period_count / char_count
    
    pause_count = neutral_text.count('，') + neutral_text.count('；') + neutral_text.count('、')
    pause_density = pause_count / char_count
    
    emotion_count = neutral_text.count('？') + neutral_text.count('！')
    emotion_density = emotion_count / char_count
    
    # 功能词密度（核心风格标记，与内容无关）
    func_count = sum(1 for w in non_ent_words if w in FUNCTION_WORDS)
    func_density = func_count / non_ent_count
    
    # 结构存在性（与内容无关）
    ph_density = ph_count / char_count
    has_leading_ph = 1.0 if original_text.strip().startswith('[') else 0.0
    
    # 超短句密度（[ENT]算3字符）
    sentences = [s.strip() for s in re.split(r'[。！？;；]', neutral_text) if s.strip()]
    ultra_short = sum(1 for s in sentences if len(s.replace('[ENT]', 'XXX')) <= 5)
    ultra_short_ratio = ultra_short / max(len(sentences), 1)
    
    # === 删除 ph_len_cv，保留8维 ===
    return np.array([
        ph_density,         # 0: 占位符密度（结构复杂度）
        has_leading_ph,      # 1: 句首占位符（正式宣告）
        ellipsis_density,    # 2: 省略号密度（迟疑/停顿）
        period_density,      # 3: 句号密度（短句宣言）
        func_density,        # 4: 功能词密度（之/而/无妨/但是...核心特征）
        pause_density,       # 5: 内部停顿（节奏控制）
        emotion_density,     # 6: 情感标点（？！强度）
        ultra_short_ratio    # 7: 超短句密度（宣言式风格）
    ], dtype=np.float64)

# 特征名称
feature_names = [
    'placeholder_density', 'leading_ph',
    'function_word_density', 'connector_density', 'modal_density',
    'ultra_short_ratio', 'pause_density', 'emotion_density'
]

print(f"特征维度：{len(feature_names)}维（控制变量：标签内容中性化）")
print("特征说明：")
print("  [0-2] 结构层：占位符密度/长度变异/句首标记")
print("  [3-5] 功能层：虚词密度/连接词/语气词（与内容无关）")
print("  [6-8] 节奏层：短句密度/停顿节奏/情感标点")

# ========== 4. 主流程 ==========

# 读取数据
train_df = pd.read_csv(TRAIN_PATH)
valid_df = pd.read_csv(VALID_PATH)
test_df = pd.read_csv(TEST_PATH)

train_texts = train_df['text'].astype(str).tolist()
valid_texts = valid_df['text'].astype(str).tolist()
all_texts = train_texts + valid_texts
test_texts = test_df['text'].astype(str).tolist()

print(f"训练+验证集：{len(all_texts)}句")
print(f"测试集：{len(test_texts)}句")

# 提取特征
print("提取训练集+验证集结构特征...")
all_features = np.array([extract_structural_features(t) for t in all_texts])
print("提取测试集结构特征...")
test_features = np.array([extract_structural_features(t) for t in test_texts])

# L2归一化
all_features_norm = normalize(all_features, norm='l2', axis=1)
test_features_norm = normalize(test_features, norm='l2', axis=1)

# 中心向量
center_vector = np.mean(all_features_norm, axis=0).reshape(1, -1)
np.save(OUTPUT_CENTER, center_vector)

# 相似度计算
similarities = cosine_similarity(test_features_norm, center_vector).flatten()
scores = similarities * 100

# 保存
results = pd.DataFrame({
    'text': test_texts,
    'o1_score': scores,
    'similarity_raw': similarities
})
results.to_csv(OUTPUT_PRED, index=False, encoding='utf-8-sig')

print(f"\n结果保存至：{OUTPUT_PRED}")
print(f"分数统计：min={scores.min():.2f}, max={scores.max():.2f}, mean={scores.mean():.2f}")

print("\n各特征均值（训练集）：")
for i, name in enumerate(feature_names):
    print(f"  {name}: {all_features[:, i].mean():.4f}")

print("\n=== O1-v2 控制变量版完成 ===")
print("控制变量验证：")
print("  - [源石] vs [概念] 统一标记为[ENT]，语义被抹平")
print("  - 功能词统计排除[ENT]token，避免内容污染")
print("  - 仅保留结构特征（密度/位置/变异）和语域标记（之/而/无妨）")
print("跑完了，去睡觉！")