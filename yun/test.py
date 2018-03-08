import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

train = pd.read_csv("data/train_first.csv")
pred = pd.read_csv('data/predict_first.csv')
xtrain,xvalid,ytrain,yvalid = train_test_split(train.Discuss.values,train.Score.values,stratify=train.Score.values,random_state=42,test_size=0.1)
print(len(train.Discuss.values))
print(train.Discuss.values[:5])
print(type(xtrain))
print(len(pred))
all = np.concatenate((xtrain,xvalid))
print(len(all))
print(xtrain[:5])
print(all[:5])
