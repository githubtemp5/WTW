#imports
from time import time
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras

from keras.models import Sequential, load_model, Model
from keras.layers import Dense, SimpleRNN, Embedding, LSTM, GRU, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import SGD
from keras import Input, layers
from random import randint
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler

__main__ = "__main__"
features = ['Year', 'Month']
output = ['Sales']
file_path = "C:\\Users\\ShresthaAl\\Documents\\Datasets\\JulyData1.csv"
models_path = "C:\\Users\\ShresthaAl\\Documents\\models\\"
batch_size_rate = 1

df = pd.read_csv(file_path)

#process data
import os

f = open(file_path)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
del lines[-1]
lines= lines[1:]
print(header)
print(len(lines))

float_data = np.zeros((len(lines), len(header)))

for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[0:]]
    print(values)
    float_data[i, :] = values
    
temp = float_data[:, 1]
#plt.plot(range(len(temp)),temp)
#plt.show()

lookback = 1
steps = 1
delay = 1

mean = float_data[:13].mean(axis=0)
float_data -= mean
std = float_data[:13].std(axis=0)
float_data /=std

def generator(data,lookback, delay, min_index, max_index, shuffle=False, batch_size=1, step=1):
    if max_index is None:
        max_index = len(data)-delay-1
    i = min_index+lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size= batch_size)
        else:
            if i+batch_size>=  max_index:
                i= min_index+lookback
            rows = np.arange(i,min(i+batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows), lookback//step, data.shape[-1]))
        
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets
        
# def evaluate_naive_method():
#     batch_maes = []
#     for step in range(val_steps):
#         samples , targets =  next(val_gen)
#         preds = samples[:, -1, 1]
#         mae = np.mean(np.abs(preds-targets))
#         batch_maes.append(mae)
#     print(np.mean(batch_maes))
# 
# evaluate_naive_method()
                


model = Sequential()
model.add(layers.LSTM(3, recurrent_dropout=0.1, input_shape=(None,float_data.shape[-1]), return_sequences=True))#,dropout=0.1
model.add(layers.LSTM(3, recurrent_dropout=0.1))#, dropout=0.1
model.add(layers.Dense(1))
model.compile(optimizer = RMSprop(), loss='mae')

#model = load_model("predict7")

def train():
    train_gen = generator(float_data, lookback = lookback, delay=delay, min_index=0 ,max_index=13, shuffle=False, step = steps, batch_size=batch_size_rate)
    
    val_gen = generator(float_data, lookback = lookback, delay = delay, min_index = 13, max_index=16, step=steps, batch_size=batch_size_rate)
    
    test_gen = generator(float_data, lookback=lookback, delay=delay, min_index = 16, max_index=17, step=steps, batch_size=batch_size_rate)
    
    val_steps = 2
    test_steps = 1
    
    history = model.fit_generator(train_gen, steps_per_epoch=100, epochs=100, validation_data = val_gen, validation_steps=val_steps, verbose=0)
    
    #model = load_model("predict6")
    
    # model = Sequential()
    # model.add(layers.Flatten(input_shape=(lookback//steps, float_data.shape[-1])))
    # model.add(layers.Dense(32, activation = 'relu'))
    # model.add(layers.Dense(1))
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.legend()
    plt.show()
    
    preds = model.predict_generator(test_gen, steps = test_steps)


    print(preds*std[1])


# #initialise model
# model = Sequential()
# model.add(GRU(units = 1, return_sequences = True, input_shape = (None,1)))
# model.add(GRU(units=5, return_sequences=True))
# model.add(GRU(units=5, return_sequences=True))
# model.add(Dense(1))
# 
# model.compile(optimizer='rmsprop', loss = 'mse', metrics = ['mae'])
# 
# 
# 
# #look up doc for sequence pad_sequence
# 
# x = df['Year']
# y = df['Existing book']
# 
# mean = x.mean(axis=0)
# x -= mean
# std = x.std(axis=0)
# x /= std
# 
# mean = y.mean(axis=0)
# y -= mean
# std = y.std(axis=0)
# y /= std
#     
# x = df['Year'].values
# x = x.reshape(17,1,1)
# y = df['Existing book'].values
# y = y.reshape(17,1,1)
# 
# df2 = pd.read_csv("C:\\Users\\ShresthaAl\\Documents\\Datasets\\predictValues.csv")
# 
# x_test = df2['Year'].values
# x_test = x_test.reshape(1,1,1)
# 
# 
# #train_x = sequence.pad_sequences(df['Year'].values.reshape(17,1), maxlen=1)
# #train_y = sequence.pad_sequences(df['Existing book'].values.reshape(17,1), maxlen=1)
# 
# history = model.fit(x,y, epochs=10, batch_size = batch_size_rate, validation_split=0.2)
# 
# preds = model.predict(x_test, steps=1)
# print('Prediction:   ')
# print(preds)
