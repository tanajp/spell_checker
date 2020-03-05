from janome.tokenizer import Tokenizer
import io

# データの読み込み
path = "data.txt"
with io.open(path, encoding="utf-8") as f:
    text = f.read().split()

# 前処理 (分かち書き)
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

# 前処理 (ベクトル化)

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

# xとyの準備
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
