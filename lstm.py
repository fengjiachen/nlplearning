from sklearn.model_selection import train_test_split
import nltk  # 用来分词
import collections  # 用来统计词频
import numpy as np
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Dense, Activation, LSTM, Embedding
from tensorflow.keras import Sequential

# from tensorflow.keras.layers.core import Activation, Dense
import pickle

MAX_FEATURES = 2000
MAX_SENTENCE_LENGTH = 40

maxlen = 0
word_freqs = collections.Counter()
num_recs = 0
with open('./data/training.txt', 'r', encoding='utf-8') as f:
    for line in f:
        label, sentence = line.strip().split("\t")
        words = nltk.word_tokenize(sentence.lower())
        if len(words) > maxlen:
            maxlen = len(words)
        for word in words:
            word_freqs[word] += 1
        num_recs += 1
print('max_len ', maxlen)
print('nb_words ', len(word_freqs))

vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
word2index = {x[0]: i+2 for i,
              x in enumerate(word_freqs.most_common(MAX_FEATURES))}
word2index["PAD"] = 0
word2index["UNK"] = 1
index2word = {v: k for k, v in word2index.items()}

X = np.empty(num_recs, dtype=list)
y = np.zeros(num_recs)
i = 0
with open('./data/training.txt', 'r+', encoding='utf-8') as f:
    for line in f:
        label, sentence = line.strip().split("\t")
        words = nltk.word_tokenize(sentence.lower())
        seqs = []
        for word in words:
            if word in word2index:
                seqs.append(word2index[word])
            else:
                seqs.append(word2index["UNK"])
        X[i] = seqs
        y[i] = int(label)
        i += 1
X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42)

EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64

model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_SIZE,
                    input_length=MAX_SENTENCE_LENGTH))
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss="binary_crossentropy",
              optimizer="adam", metrics=["accuracy"])

BATCH_SIZE = 512
NUM_EPOCHS = 20
model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE,
          epochs=NUM_EPOCHS, validation_data=(Xtest, ytest))

pre = model.predict(Xtest)

for i in range(len(pre)):
    print(Xtest[i], ytest[i], pre[i])
    
