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

__main__ = "__main__"
features = ['Year', 'Month']
output = ['Sales']
file_path = "C:\\Users\\ShresthaAl\\Documents\\Datasets\\PMI_dataset_large_updated_lossBusiness.csv"
models_path = "C:\\Users\\ShresthaAl\\Documents\\models\\"
learning_rate = 0.015
epoch_rate = 50
batch_size_rate = 10


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

lookback = 20
steps = 1
delay=1

mean = float_data[:170].mean(axis=0)
float_data -= mean
std = float_data[:170].std(axis=0)
float_data /=std

def generator(float_data,lookback, delay, min_index, max_index, shuffle=False, batch_size=1, step=1):
    if max_index is None:
        max_index = len(float_data)-delay-1
    i = min_index+lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size= batch_size)
        else:
            if i+batch_size>=  max_index:
                i= min_index+lookback
            rows = np.arange(i,min(i+batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows), lookback//step, float_data.shape[-1]))
        
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = float_data[indices]
            targets[j] = float_data[rows[j] + delay][1]
        yield samples, targets
                

train_gen = generator(float_data, lookback = lookback, delay=delay, min_index=0,max_index=170, shuffle=True, step=steps, batch_size=batch_size_rate)

val_gen = generator(float_data, lookback = lookback, delay = delay, min_index = 171, max_index=208, step=steps, batch_size=batch_size_rate)

test_gen = generator(float_data, lookback=lookback, delay=delay, min_index = 171, max_index=None, step=steps, batch_size=batch_size_rate)

val_steps = (208-171-lookback)
test_steps = (len(float_data) - 208-lookback)

model = Sequential()
model.add(layers.Flatten(input_shape=(lookback//steps, float_data.shape[-1])))
model.add(layers.Dense(32, activation = 'relu'))
model.add(layers.Dense(1))

model.compile(optimizer = RMSprop(), loss='mae')
history = model.fit_generator(train_gen, steps_per_epoch=100, epochs=20,                 validation_data = val_gen, validation_steps=val_steps)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()

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
