"""
LSTM HALF WORKING
"""
from time import time
import os
from random import shuffle

import numpy
import pandas

from keras import Input, layers
from keras.models import Sequential, Model
from keras.layers import LSTM

# Tell keras to use tensorflow as back-end
os.environ['KERAS_BACKEND'] = 'tensorflow'

import pandas as pd
import seaborn as sns

from keras.callbacks import TensorBoard
from keras.optimizers import SGD, RMSprop

df = pd.read_csv('C:\\Users\\ChoudhuryMB\\Desktop\\keras-work\\using_dataset\\lstm.csv',
                 header=None,
                 sep=',')

df.columns = ['year', 'month', 'sales']

df = df.values


def select_features(df):
    return df[:, 1]

def select_label(df):
    return df[:, 2]

feature = select_features(df)
label = select_label(df)

print(len(feature), len(label))

feature = feature.reshape(215, 1, 1)

print(feature)

def make_model():
    model = Sequential()
    model.add(LSTM(8, input_shape=(None, 1)))
    model.add(layers.Dense(1))
    
    model.summary()
    
    return model
print("Creating model")

model = make_model()


print("Compiling model")
model.compile(optimizer=RMSprop(),
             loss='mse',
             metrics=['mae'])

tensorboard = TensorBoard(log_dir='C:\\Users\\ChoudhuryMB\\Desktop\\keras-work\\test1\\{}'.format(time()))

print("Starting to train...",
      "(Check TensorBoard for Training Loss and Validation loss)")
summary = model.fit(
        feature, label,
        epochs=80,
        verbose=1,
        callbacks=[tensorboard])


#score = model.evaluate(test_examples, test_targets, verbose=0)

#print('Test loss:', (score[0]))
#print('Test accuracy:', (score[1]))

def outcome_pred(index):
    pred = pd.read_csv("C:\\Users\\ChoudhuryMB\\Desktop\\book1.csv",
                       header=None,
                       sep=',')
    pred.columns = ['year', 'months', 'existing_book', 'new_business', 'lost_business', 'sales']
    print("Prediction:")
    predicts = model.predict(pred.iloc[index:index+1]['year'], verbose=0, steps=10)
    print(predicts[:][0])
