import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')


from keras.layers.core import Dense, Activation, Dropout
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils

from keras.layers import GlobalAveragePooling1D, Embedding
from keras.callbacks import EarlyStopping
from keras.layers.recurrent import LSTM, GRU
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
import jieba
from keras.layers import Activation

import numpy as np
import os
import sys
import random
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Activation

import yaml
from keras.models import model_from_yaml


train = pd.read_csv("data/train_first.csv")
pred = pd.read_csv('data/predict_first.csv')
print(train.head())

def multiclass_logloss(actual,predicted,eps=1e-15):
    if len(actual.shape)==1:
        actual2 = np.zeros(actual.shape[0],predicted.shape[1])
        for i,val in enumerate(actual):
            actual2[i,val] = 1
        actual = actual2

    clip = np.clip(predicted,eps,1-eps)
    rows = actual.shape[0]
    vsota = np.sum(actual*np.log(clip))
    return -1.0 / rows * vsota


print(type(train.Score.values[0]))
print(train.Score.values)

xtrain,xvalid,ytrain,yvalid = train_test_split(train.Discuss.values,train.Score.values,stratify=train.Score.values,random_state=42,test_size=0.1)
xpred = pred.Discuss.values
embeddings_index = {}
f = open('wiki.zh.jian.stop.text.vector', 'r', encoding='utf-8')
for line in tqdm(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
tokenizer = Tokenizer()
tokenizer.fit_on_texts(xtrain)
print(xtrain)


# this function creates a normalized vector for the whole sentence

ytrain_enc = np_utils.to_categorical(ytrain)
yvalid_enc = np_utils.to_categorical(yvalid)

word_index = {}
#embedding_matrix=np.arange()
count = {}
all_sentences = np.concatenate((train.Discuss.values,pred.Discuss.values))

for words in all_sentences:
    #print(type(words))
    words = jieba.cut(words)
    words = ' '.join(words)
    words = words.split(' ')
    #print(words)
    #words = list(words)
    #print(words)
    for w in words:
        count[w]=w

print(len(count))
ind=[]
embedding_matrix = np.zeros((100497,300))
X = []
co = {}
for words in xtrain:
    words = jieba.cut(words)
    words = ' '.join(words)
    words = words.split(' ')
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    #X.append(words)
    M = []
    for w in words:
        try:
            if w not in co:
                ind.append(w)
                co[w] = w
                i = ind.index(w)
                embedding_matrix[i] = embeddings_index[w]
                word_index[w]=i
            M.append(word_index[w])
                #print(M)
        except:
            continue
    X.append(M)

V = []
for words in xvalid:
    words = jieba.cut(words)
    words = ' '.join(words)
    words = words.split(' ')
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    #X.append(words)
    M = []
    for w in words:
        try:
            if w not in co:
                ind.append(w)
                co[w] = w
                i = ind.index(w)
                embedding_matrix[i] = embeddings_index[w]
                word_index[w]=i
            M.append(word_index[w])
                #print(M)
        except:
            continue
    V.append(M)

P = []
for words in xpred:
    words = jieba.cut(words)
    words = ' '.join(words)
    words = words.split(' ')
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    #X.append(words)
    M = []
    for w in words:
        try:
            if w not in co:
                ind.append(w)
                co[w] = w
                i = ind.index(w)
                embedding_matrix[i] = embeddings_index[w]
                word_index[w]=i
            M.append(word_index[w])
                #print(M)
        except:
            continue
    P.append(M)

print(word_index)

data = pad_sequences(X, maxlen=300)
vdata = pad_sequences(V,maxlen = 300)
pdata = pad_sequences(P,maxlen = 300)
# print('Preparing embedding matrix.')
#
# nb_words = min(MAX_NB_WORDS, len(word_index))
# #20000
# embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
#
# for word, i in word_index.items():
#     if i > MAX_NB_WORDS:
#         continue
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         # words not found in embedding index will be all-zeros.
#         embedding_matrix[i] = embedding_vector
#
# print(embedding_matrix.shape)



model = Sequential()

model.add(Embedding(100497,
                     300,
                     weights=[embedding_matrix],
                     input_length=300,
                     trainable=False))
#model.add(Embedding(6000,300,input_length=300,trainable=False))
# model.add(Dense(300, input_dim=300, activation='relu'))

model.add(LSTM(50, activation='sigmoid', inner_activation='hard_sigmoid'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(6))
model.add(Activation('softmax'))

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

model.fit(data, y=ytrain_enc, batch_size=32,
          epochs=10, verbose=1,
          validation_data=(vdata, yvalid_enc))

score, acc = model.evaluate(vdata, yvalid_enc,batch_size=128)
print('Test score:', score)
print('Test accuracy:', acc)

target_vars=[1,2,3,4,5]
print(model.predict(pdata))

predicts = model.predict(pdata)

score = []

for pre in predicts:
    pre = pre.tolist()
    pre =pre.index(max(pre))
    score.append(pre)

print(score)
preds = pd.DataFrame(score)
submission = pd.concat([pred['Id'],preds], 1)
submission.to_csv("./2submission.csv", index=False)
submission.head()

yaml_string = model.to_yaml()
with open('lstm_data/lstm.yml', 'w') as outfile:
    outfile.write(yaml.dump(yaml_string, default_flow_style=True))
model.save_weights('lstm_data/lstm.h5') 