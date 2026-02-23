import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import jieba
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

# === 路径配置（自动计算，无需修改） ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'b4')
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'final', 'train.csv')
VALID_PATH = os.path.join(BASE_DIR, 'data', 'final', 'validation.csv')
TEST_PATH = os.path.join(BASE_DIR, 'data', 'test', 'test_positive.csv')

OUTPUT_PRED = os.path.join(RESULTS_DIR, 'predictions.csv')
OUTPUT_MODEL = os.path.join(RESULTS_DIR, 'word2vec.model')
OUTPUT_CENTER = os.path.join(RESULTS_DIR, 'center_vector.npy')

# === 训练监控器（每个epoch打印loss）===
class LossPrinter(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.losses = []  # 记录历史，方便后续画图
    
    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        self.losses.append(loss)
        self.epoch += 1
        print(f"  Epoch {self.epoch}/{EPOCHS} - Loss: {loss:.4f}")

# 实例化
loss_printer = LossPrinter()

# === Word2Vec 参数（可按需调整） ===
VECTOR_SIZE = 50      # 词向量维度
WINDOW = 2             # 上下文窗口
MIN_COUNT = 1          # 最小词频（保留所有词）
SG = 1                 # 0=CBOW, 1=Skip-gram
EPOCHS = 50            # 训练轮数
WORKERS = 1            # 线程数
SEED = 325             # 已完成今日325大学习

print("=== B4 Word2Vec 基线（有神经网络，但CPU也能跑） ===")

# 1. 读取数据
train_df = pd.read_csv(TRAIN_PATH)
valid_df = pd.read_csv(VALID_PATH)
test_df = pd.read_csv(TEST_PATH)

train_texts = train_df['text'].astype(str).tolist()
valid_texts = valid_df['text'].astype(str).tolist()
all_texts = train_texts + valid_texts        # 合并训练和验证作为训练语料
test_texts = test_df['text'].astype(str).tolist()

print(f"训练语料（train+valid）: {len(all_texts)} 句")
print(f"测试集: {len(test_texts)} 句")

# 2. 分词（使用 jieba）
print("分词中...")
def tokenize(texts):
    return [jieba.lcut(text.strip()) for text in texts]

train_tokenized = tokenize(all_texts)
test_tokenized = tokenize(test_texts)

# 3. 训练 Word2Vec 模型
print("训练 Word2Vec 模型...")
model = Word2Vec(
    sentences=train_tokenized,
    vector_size=VECTOR_SIZE,
    window=WINDOW,
    min_count=MIN_COUNT,
    sg=SG,
    epochs=EPOCHS,
    workers=WORKERS,
    seed=SEED,
    compute_loss=True,
    callbacks=[loss_printer]   
)
print(f"词表大小: {len(model.wv.key_to_index)}")
# 训练完成后打印趋势（放在 model.save 之前）
print(f"\nLoss趋势：{loss_printer.losses[0]:.2f} → {loss_printer.losses[-1]:.2f}")
if len(loss_printer.losses) > 1 and loss_printer.losses[-1] < loss_printer.losses[0]:
    print("模型已收敛")
else:
    print("注意：loss无明显下降（可能语料过短或已收敛）")

# 4. 定义句子向量函数（平均词向量）
def sentence_vector(words, model, vector_size):
    vectors = [model.wv[word] for word in words if word in model.wv]
    if not vectors:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)

# 5. 计算训练集句子向量及中心向量
print("计算句子向量...")
train_vectors = np.array([
    sentence_vector(words, model, VECTOR_SIZE) for words in train_tokenized
])
center_vector = np.mean(train_vectors, axis=0).reshape(1, -1)
print("中心向量计算完成")

# 6. 测试集预测
print("预测测试集...")
test_vectors = np.array([
    sentence_vector(words, model, VECTOR_SIZE) for words in test_tokenized
])
similarities = cosine_similarity(test_vectors, center_vector).flatten()
scores = similarities * 100

# 7. 保存结果
print("保存结果...")
model.save(OUTPUT_MODEL)
np.save(OUTPUT_CENTER, center_vector)

results = pd.DataFrame({
    'text': test_texts,
    'b4_score': scores,
    'similarity_raw': similarities
})
results.to_csv(OUTPUT_PRED, index=False, encoding='utf-8-sig')

print(f"\n结果保存至：{RESULTS_DIR}")
print(f"分数统计：min={scores.min():.2f}, max={scores.max():.2f}, mean={scores.mean():.2f}")
print("=== 完成 ===")