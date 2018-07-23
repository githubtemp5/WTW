import os
from time import time

import numpy
import pandas

import keras
from keras.models import Sequential
from keras import layers, optimizers


# Tell keras to use tensorflow as back-end
os.environ['KERAS_BACKEND'] = 'tensorflow'

import inspect

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from keras.callbacks import TensorBoard

#from utils import plot_training_summary
#from utils import TimeSummary
#from utils import set_seed

moons = pd.read_csv('C:\\Users\\choudhuryMB\\Desktop\\data\\moons.csv')
print('(rows, columns):', moons.shape)
moons.sample(3)

sns.lmplot(data=moons, x='x1', y='x2', hue='y', fit_reg=False);

train_index = moons.sample(frac=0.8, random_state=21).index
X_train, y_train = moons.iloc[train_index][['x1', 'x2']], moons.iloc[train_index]['y']
X_test, y_test = moons.drop(index=train_index)[['x1', 'x2']], moons.drop(index=train_index)['y']


def make_model():
    model = Sequential(name='keras-model-1')
    
    # input layer
    model.add(layers.Dense(name='input-HL_1', units=3, input_dim=2))
    model.add(layers.Activation(name='ReLu_1', activation='relu'))
    
    # hidden layer 1
    model.add(layers.Dense(name='HL_2', units=2))
    model.add(layers.Activation(name='ReLu_2', activation='relu'))
    
    # output layer
    model.add(layers.Dense(name='Output', units=1))
    model.add(layers.Activation(name='Sigmoid_3', activation='sigmoid'))
    
    model.summary()

    return model

model = make_model()

model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Runned on TB")
tensorboard = TensorBoard(log_dir='C:\\Users\\choudhuryMB\\Desktop\\data\\{}'.format(time()))

print("Starting...")
summary = model.fit(
        X_train, y_train,
        batch_size=1,
        epochs=50,
        validation_split=0.1,
        verbose=0,
        callbacks=[tensorboard])

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
