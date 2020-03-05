from janome.tokenizer import Tokenizer

# データの読み込み
path = "data.txt"
file = open(path, "r", encoding="utf-8")
text = file.read()

# 前処理 (分かち書き)
t = Tokenizer()
def wakati(text):
    w = t.tokenize(text, wakati=True)
    return " ".join(w)

data = [wakati(w) for w in data]

import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional

# モデルから文字列を生成。
def generate_seq(model, tokenizer, seed_text, n_words):
    in_text, result = seed_text, seed_text
    for _ in range(n_words):
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        encoded = np.array(encoded)
        # 学習データから文字列を予測。
        yhat = model.predict_classes(encoded, verbose=0)
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        in_text, result = out_word, result + ' ' + out_word
    return result

# 前処理 (ベクトル化)
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
encoded = tokenizer.texts_to_sequences([data])[0]

# 語彙のサイズを決定。
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

sequences = list()
for i in range(1, len(encoded)):
    sequence = encoded[i-1:i+1]
    sequences.append(sequence)
print('Total Sequences: %d' % len(sequences))

# Xとy要素に分割する.
sequences = np.array(sequences)
X, y = sequences[:,0],sequences[:,1]

# One-hotエンコーディング
y = to_categorical(y, num_classes=vocab_size)

# モデリング
model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=1))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 学習させる.
model.fit(X, y, epochs=500, verbose=2)

# 開始地点を決定.
typo_words = []
typo_index = []
if len(words) != len(sentence):
    for i in range(len(sentence)):
        if not str(sentence[i]) in words:
            typo_words.append(str(sentence[i]))
            typo_index.append(sentence.index(typo_words[0]))

search_index = []
search_word = []
for i in range(len(typo_index)):
    search_index.append(int(typo_index[i])-1)
    search_word.append(sentence[search_index[i]])

# 評価
for i in range(len(search_word)):
    print("prediction: " + generate_seq(model, tokenizer, str(search_word[i]) , 2))
