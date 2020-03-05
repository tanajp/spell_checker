# スペルチェッカー

<br>

## 概要
Bidirectional LSTMによる誤字脱字、衍字の検出、置き換え。    
確率モデルによって誤りを検出し、言語モデルによってそれを候補文字で置き換え生成する。(試作品)

<br>

## ソースコード

### 確率モデル

```python
from janome.tokenizer import Tokenizer

#前処理

data = [
    "100名まで収容可能な会場。",
    "ドレスのご試着は、",
    "ご要望にお応えします。",
    "写真撮影を行います。",
    "宜しくお願いします。"
]

t = Tokenizer()
def wakati(text):
    w = t.tokenize(text, wakati=True)
    return " ".join(w)

data = [wakati(w) for w in data]

from tensorflow import keras
from keras.preprocessing import sequence
from keras import preprocessing
import keras
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional

# 前処理

tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)
vocab = tokenizer.word_index
seqs = tokenizer.texts_to_sequences(data)

# シーケンスを同じ長さになるように詰める.
def prepare_sentence(seq, maxlen):
    x = []
    y = []
    for i, w in enumerate(seq):
        x_padded = pad_sequences([seq[:i]],
                                 maxlen=maxlen - 1,
                                 padding='pre')[0]
        x.append(x_padded)
        y.append(w)
    return x, y

maxlen = max([len(seq) for seq in seqs])
x = []
y = []
for seq in seqs:
    x_windows, y_windows = prepare_sentence(seq, maxlen)
    x += x_windows
    y += y_windows
x = np.array(x)
y = np.array(y) - 1
y = np.eye(len(vocab))[y]

# モデリング
model = Sequential()
model.add(Embedding(input_dim=len(vocab) + 1,  
                    output_dim=5,
                    input_length=maxlen - 1)) 
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(len(vocab), activation='softmax'))

model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

# 学習させる.
model.fit(x, y, epochs=1000)

# 発生確率を計算する.
input_sentence = "100名まで収容可能な海上。"
sentence = t.tokenize(input_sentence, wakati=True)
tok = tokenizer.texts_to_sequences([sentence])[0]
x_test, y_test = prepare_sentence(tok, maxlen)
x_test = np.array(x_test)
y_test = np.array(y_test) - 1 
p_pred = model.predict(x_test)  
vocab_inv = {v: k for k, v in vocab.items()}

# OK/NG確率に基づく正常or異常の判定.
log_p_sentence = 0
err = []
words = []
for i, prob in enumerate(p_pred):
    word = vocab_inv[y_test[i]+1]
    words.append(word)
    history = ' '.join([vocab_inv[w] for w in x_test[i, :] if w != 0])
    prob_word = prob[y_test[i]]
    log_p_sentence += np.log(prob_word)
    
    if prob_word < 0.03:
        err.append(word)

    print('P(w={}|h={})={}'.format(word, history, prob_word))
print('Prob. sentence: {}'.format(np.exp(log_p_sentence)))

# 「誤字脱字箇所」と「誤字脱字を含む文」を出力.
if len(err) != 0:
    print("NG : " + str(err))
    print(input_sentence)

# 訂正箇所なしの場合
else:
    print("OK")
```
***

### 言語モデル

```python
# 前処理
from janome.tokenizer import Tokenizer

data = """
    100名まで収容可能な会場。\n
    ドレスのご試着は、\n
    ご要望にお応えします。\n
    写真撮影を行います。\n
    宜しくお願いします。\n
"""

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
```
