#imports
from time import time
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, SimpleRNN, Embedding, LSTM
from keras.callbacks import TensorBoard
from keras.optimizers import SGD
from keras import Input, layers
from random import randint
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence

__main__ = "__main__"
features = ['Year', 'Month']
output = ['Sales']
file_path = "C:\\Users\\ShresthaAl\\Documents\\Datasets\\AugustData.csv"
models_path = "C:\\Users\\ShresthaAl\\Documents\\models\\"
learning_rate = 0.015
epoch_rate = 70
batch_size_rate = 10


df = pd.read_csv(file_path)

model = Sequential()
model.add(SimpleRNN(units = 1, return_sequences = True, input_shape = (None,1)))
model.add(SimpleRNN(units=5, return_sequences=True))
model.add(SimpleRNN(units=5, return_sequences=True))
model.add(Dense(1))

model.compile(optimizer='rmsprop', loss = 'mse', metrics = ['mae'])

#look up doc for sequence pad_sequence

x = df['Year'].values
x = x.reshape(17,1,1)
y = df['Existing book'].values
y = y.reshape(17,1,1)

df2 = pd.read_csv("C:\\Users\\ShresthaAl\\Documents\\Datasets\\predictValues.csv")

x_test = df2['Year'].values
x_test = x.reshape(17,1,1)

#train_x = sequence.pad_sequences(df['Year'].values.reshape(17,1), maxlen=1)
#train_y = sequence.pad_sequences(df['Existing book'].values.reshape(17,1), maxlen=1)

history = model.fit(x,y, epochs=10, batch_size = batch_size_rate, validation_split=0.2)

preds = model.predict(x_test, steps=1)
print('Prediction:   ')
print(preds)
