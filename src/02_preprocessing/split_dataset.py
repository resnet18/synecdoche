import pandas as pd
from sklearn.model_selection import train_test_split
import os

# === 配置区 ===
RANDOM_SEED = 325 #已完成今日325大学习
TEST_SIZE = 70
ZERO_SHOT_SIZE = 15
LONG_THRESHOLD = 35

# === 路径配置 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_PATH = os.path.join(BASE_DIR, 'data', 'final', 'logos_desensitized.csv')
FINAL_DIR = os.path.join(BASE_DIR, 'data', 'final')
TEST_DIR = os.path.join(BASE_DIR, 'data', 'test')

os.makedirs(FINAL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# 1. 读取数据
df = pd.read_csv(INPUT_PATH)
df['text'] = df['text'].astype(str)
df['length'] = df['text'].apply(len)
df['is_long'] = df['length'] > LONG_THRESHOLD

# 2. 抽测试集正例（给人打分用，模型绝对不看）
long_df = df[df['is_long'] == True]
short_df = df[df['is_long'] == False]
n_long_test = min(TEST_SIZE // 7, len(long_df))

test_positive = pd.concat([
    long_df.sample(n=n_long_test, random_state=RANDOM_SEED),
    short_df.sample(n=TEST_SIZE - n_long_test, random_state=RANDOM_SEED)
])
test_positive['label'] = 1
test_positive['source'] = 'logos'

# 3. 划分训练集/验证集（O2用）
remaining = df.drop(test_positive.index)
train, val = train_test_split(
    remaining, test_size=0.2, random_state=RANDOM_SEED, stratify=remaining['is_long']
)
train['label'] = 1
val['label'] = 1
train['source'] = 'logos'
val['source'] = 'logos'

# 4. 生成B6/B7的参考示例（从训练集抽，无排序，无标注）
train_long = train[train['is_long'] == True]
train_short = train[train['is_long'] == False]
train_medium = train[(train['length'] > 20) & (train['length'] <= LONG_THRESHOLD)]

n_per_layer = (ZERO_SHOT_SIZE // 3) + 1
zero_shot_samples = []
for layer_df in [train_short, train_medium, train_long]:
    if len(layer_df) > 0:
        zero_shot_samples.append(layer_df.sample(n=min(n_per_layer, len(layer_df)), random_state=RANDOM_SEED))

zero_shot = pd.concat(zero_shot_samples).sample(frac=1, random_state=RANDOM_SEED).head(ZERO_SHOT_SIZE)
zero_shot_clean = zero_shot[['id', 'text']].reset_index(drop=True)

# 5. 测试集负例占位符
max_id = df['id'].max() if 'id' in df.columns else 0 # 在构建 negative_template 前获取最大 id（可选）

negative_template = pd.DataFrame({
    'id': [-(i+1) for i in range(TEST_SIZE)],          # 负 id，从 -1 开始递减
    'text': [f'[NEGATIVE_PLACEHOLDER_{i}]' for i in range(TEST_SIZE)],
    'length': [0] * TEST_SIZE,
    'is_long': [False] * TEST_SIZE,
    'label': [0] * TEST_SIZE,
    'source': ['to_be_filled'] * TEST_SIZE
})

test_full = pd.concat([test_positive, negative_template]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# 6. 保存
train.to_csv(os.path.join(FINAL_DIR, 'train.csv'), index=False, encoding='utf-8-sig')
val.to_csv(os.path.join(FINAL_DIR, 'validation.csv'), index=False, encoding='utf-8-sig')
test_full.to_csv(os.path.join(TEST_DIR, 'test.csv'), index=False, encoding='utf-8-sig')
test_positive.to_csv(os.path.join(TEST_DIR, 'test_positive.csv'), index=False, encoding='utf-8-sig')
zero_shot_clean.to_csv(os.path.join(FINAL_DIR, 'reference.csv'), index=False, encoding='utf-8-sig')
negative_template.to_csv(os.path.join(TEST_DIR, 'test_negative_template.csv'), index=False, encoding='utf-8-sig')

print(f"已完成划分，训练集：{len(train)}， 验证集：{len(val)}，测试集正例: {len(test_positive)}，B6/B7示例：{len(zero_shot_clean)}")