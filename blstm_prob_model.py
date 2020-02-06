from tensorflow import keras
from keras.preprocessing import sequence
from keras import preprocessing
import keras
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional


data = [
    "100 名 まで 収容 可能 な 会場 。",
    "ドレス の ご 試着 は 、",
    "ご 要望 に お 応え し ます 。",
    "写真 撮影 を 行い ます 。",
    "宜しく お 願い し ます 。"
]

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
sentence = "ドレス の を ご 試着 は 、"
tok = tokenizer.texts_to_sequences([sentence])[0]
x_test, y_test = prepare_sentence(tok, maxlen)
x_test = np.array(x_test)
y_test = np.array(y_test) - 1 
p_pred = model.predict(x_test)  
vocab_inv = {v: k for k, v in vocab.items()}

# OK/NG確率に基づく正常or異常の判定.
log_p_sentence = 0
err = []
for i, prob in enumerate(p_pred):
    word = vocab_inv[y_test[i]+1]
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
    print(sentence)

else:
    print("OK")

