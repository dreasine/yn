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

def loadfile():
    train = pd.read_csv("train_first.csv")
    pred = pd.read_csv('predict_first.csv')
