import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding

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

data = """
    100 名 まで 収容 可能 な 会場 。\n
    ドレス の ご 試着 は 、\n
    ご 要望 に お 応え し ます 。\n
    写真 撮影 を 行い ます 。\n
    宜しく お 願い し ます 。\n
"""

# 前処理
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
    print("prediction: " + generate_seq(model, tokenizer, str(search_word[i]) , 1))