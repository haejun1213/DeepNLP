import pandas as pd
import numpy as np
import os
import re
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop
import joblib

# ----------------------------
# 불용어 & 형태소 분석 준비
# ----------------------------
stopwords = set([
    '이', '그', '저', '것', '들', '의', '가', '은', '는', '에', '와', '한', '하다', '을', '를',
    '으로', '도', '으로서', '으로써', '그리고', '그러나', '하지만', '때문에', '더', '에서', '해서'
])
okt = Okt()

def preprocess_text(text):
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", "", text)
    tokens = okt.pos(text, stem=True)
    filtered = [word for word, pos in tokens if pos in ['Noun', 'Verb', 'Adjective'] and word not in stopwords]
    return ' '.join(filtered)

# ----------------------------
# 데이터 불러오기 및 전처리
# ----------------------------
review_df = pd.read_csv('C:/Lec/_DeepNLP25/Crawler/Yes24Crawler/yes24_reviews_dataset_balanced8.csv')
review_df = review_df.dropna(subset=['review'])

# 클래스 균형 맞추기
pos_df = review_df[review_df['sentiment'] == 'positive']
neu_df = review_df[review_df['sentiment'] == 'neutral']
neg_df = review_df[review_df['sentiment'] == 'negative']
min_count = min(len(pos_df), len(neu_df), len(neg_df))
pos_df = resample(pos_df, replace=False, n_samples=min_count, random_state=42)
neu_df = resample(neu_df, replace=False, n_samples=min_count, random_state=42)
neg_df = resample(neg_df, replace=False, n_samples=min_count, random_state=42)
balanced_df = pd.concat([pos_df, neu_df, neg_df]).sample(frac=1).reset_index(drop=True)

# 리뷰 정제
print(" 리뷰 전처리 중...")
balanced_df['clean_review'] = balanced_df['review'].apply(preprocess_text)

# 라벨 인코딩
label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
balanced_df['label'] = balanced_df['sentiment'].map(label_map)
input_list = list(balanced_df['clean_review'])
label_list = list(balanced_df['label'])

# 훈련/테스트 분할
X_train_text, X_test_text, y_train, y_test = train_test_split(
    input_list, label_list, test_size=0.1, stratify=label_list, random_state=42)

# ----------------------------
# 토크나이저 및 시퀀싱
# ----------------------------
vocab_size = 25000
tokenizer = Tokenizer(num_words=vocab_size + 1, oov_token="<OOV>")
tokenizer.fit_on_texts(input_list)

train_sequences = tokenizer.texts_to_sequences(X_train_text)
test_sequences = tokenizer.texts_to_sequences(X_test_text)

# 빈 시퀀스 제거
filtered_train = [(seq, label) for seq, label in zip(train_sequences, y_train) if len(seq) > 0]
filtered_test = [(seq, label) for seq, label in zip(test_sequences, y_test) if len(seq) > 0]
train_sequences, y_train = zip(*filtered_train)
test_sequences, y_test = zip(*filtered_test)
train_sequences = list(train_sequences)
test_sequences = list(test_sequences)
y_train = list(y_train)
y_test = list(y_test)

# 패딩
max_len = 50
X_train = pad_sequences(train_sequences, maxlen=max_len, padding='post', truncating='post')
X_test = pad_sequences(test_sequences, maxlen=max_len, padding='post', truncating='post')

# 원-핫 인코딩
y_train = to_categorical(y_train, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)

# ----------------------------
# 클래스 가중치 계산
# ----------------------------
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(label_list),
                                     y=label_list)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
print("📊 클래스 가중치:", class_weight_dict)

# ----------------------------
# 모델 정의
# ----------------------------
model = Sequential([
    Embedding(input_dim=vocab_size + 1, output_dim=64, input_length=max_len),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dense(64, activation='tanh'),
    Dense(3, activation='softmax')
])
model.compile(optimizer=RMSprop(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ----------------------------
# 콜백 및 학습
# ----------------------------
os.makedirs('./model', exist_ok=True)
checkpoint_path = './model/best_model_lstm_okt_imp.h5'
es = EarlyStopping(monitor='val_accuracy', mode='max', patience=3, verbose=1)
mc = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

history = model.fit(X_train, y_train,
                    epochs=20,
                    batch_size=128,
                    validation_split=0.1,
                    callbacks=[es, mc],
                    class_weight=class_weight_dict)

# ----------------------------
# 평가 및 저장
# ----------------------------
model.load_weights(checkpoint_path)
loss, acc = model.evaluate(X_test, y_test)
print(f"\n🧪 최종 테스트 정확도: {acc * 100:.2f}%")

joblib.dump(tokenizer, "./model/yes24_tokenizer.pkl")
