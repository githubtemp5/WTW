"""
REGRESSION MODEL KERAS
"""
from time import time
import os

import numpy
import pandas

import keras
from keras.models import Sequential
from keras import layers, optimizers

# Tell keras to use tensorflow as back-end
os.environ['KERAS_BACKEND'] = 'tensorflow'

import pandas as pd
import seaborn as sns

from keras.callbacks import TensorBoard
from keras.optimizers import SGD

df = pd.read_csv('C:\\Users\\ChoudhuryMB\\Desktop\\keras-work\\test4.csv',
                 header=None,
                 sep=',')

df.columns = ['year', 'months', 'existing_book', 'new_business', 'lost_business', 'sales']

print("Data Description")
df.describe()


def preprocess_features(df):
    selected_features = df[['year', 'months', 'existing_book']]
    mean = selected_features.mean(axis=0)
    selected_features -= mean
    std = selected_features.std(axis=0)
    selected_features /= std
    return selected_features

def preprocess_targets(df):
    output_target = df['sales']
    mean = output_target.mean(axis=0)
    output_target -= mean
    std = output_target.std(axis=0)
    output_target /= std
    return output_target


training_examples = preprocess_features(df.head(190))
training_targets = preprocess_targets(df.head(190))

test_examples = preprocess_features(df.tail(30))
test_targets = preprocess_targets(df.tail(30))

print("Training examples summary:")
print(training_examples.describe())
print("Validation examples summary:")
print(test_examples.describe())
print("Training targets summary:")
print(training_targets.describe())
print("Validation targets summary:")
print(test_targets.describe())

#sns.lmplot(data=df, x='year', y='existing_book', hue='pred', fit_reg=False);

def make_model():
    model = Sequential(name='Revenue predicition based on years')
    
    model.add(layers.Dense(name='1', units=3, input_dim=3))
    model.add(layers.Activation(name='relu1', activation='relu'))
    
    model.add(layers.Dense(name='1x', units=2))
    model.add(layers.Activation(name='1xb', activation='relu'))
    
    model.add(layers.Dense(name='4', units=1))
    
    model.summary()
    
    return model

print("Creating model")
model = make_model()



print("Compiling model")
model.compile(optimizer=SGD(0.01),
             loss='mse',
             metrics=['mse'])

tensorboard = TensorBoard(log_dir='C:\\Users\\ChoudhuryMB\\Desktop\\keras-work\\test1\\{}'.format(time()))

print("Starting to train...",
      "(Check TensorBoard for Training Loss and Validation loss)")
summary = model.fit(
        training_examples, training_targets,
        batch_size=3,
        epochs=100,
        validation_split=0.1,
        verbose=0,
        callbacks=[tensorboard])


score = model.evaluate(test_examples, test_targets, verbose=0)

print('Test loss:', (score[0]))
print('Test accuracy:', (score[1]))

def predict(index):
    pred = pd.read_csv("C:\\Users\\ChoudhuryMB\\Desktop\\keras-work\\test4.csv",
                       header=None,
                       sep=',')
    pred.columns = ['year', 'months', 'existing_book', 'new_business', 'lost_business', 'sales']
    print("Prediction:")
    predicts = model.predict(pred.iloc[index:index+1][['year', 'months', 'existing_book']], verbose=0, steps=10)
    print(predicts[:][0])
