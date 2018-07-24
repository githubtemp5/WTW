"""
REGRESSION MODEL KERAS




NEEDS TO NORMALIZE DATA!!!



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

df = pd.read_csv('C:\\Users\\ChoudhuryMB\\Desktop\\keras-work\\data\\PMI_dataset_large.csv',
                 header=None,
                 sep=',')

df.columns = ['year', 'month', 'existing_book', 'new_business', 'lost_business', 'sales']

print("Data Description")
df.describe()

def preprocess_features(df):
    selected_features = df[['year', 'existing_book']]
    return selected_features

def preprocess_targets(df):
    output_target = df['sales']
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
    
    model.add(layers.Dense(name='1', units=2, input_dim=1))
    model.add(layers.Activation(name='relu1', activation='relu'))
    
    model.add(layers.Dense(name='2', units=2))
    model.add(layers.Activation(name='relu2', activation='relu'))
    
    model.add(layers.Dense(name='3', units=1))
    #model.add(layers.Activation(name='sigmoid', activation='sigmoid'))
    
    model.summary()
    
    return model

model = make_model()

model.compile(optimizer='rmsprop',
             loss='mse',
             metrics=['mae'])

print("Check TensorBoard for Training Loss and Validation loss")
tensorboard = TensorBoard(log_dir='C:\\Users\\ChoudhuryMB\\Desktop\\keras-work\\test1\\{}'.format(time()))

print("Starting to train the model...")
summary = model.fit(
        training_examples, training_targets,
        batch_size=1,
        epochs=50,
        validation_split=0.1,
        verbose=0,
        callbacks=[tensorboard])

score = model.evaluate(test_examples, test_targets, verbose=0)

print('Test loss:', (score[0]))
print('Test accuracy:', (score[1]))
