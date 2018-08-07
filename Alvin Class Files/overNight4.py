import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import math
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNet
import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, SimpleRNN, Embedding, LSTM, GRU, Flatten, Input
from keras.callbacks import TensorBoard
from keras.optimizers import RMSprop, Adam
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("C:\\Users\\ShresthaAl\\Documents\\Datasets\\PMI_dataset_large_updated_lossBusiness.csv", dtype={'Year':str, 'Month':str})

df2 = pd.read_csv("C:\\Users\\ShresthaAl\\Documents\\Datasets\\PMI_dataset_large_updated_lossBusiness3.csv")
print(df.describe())

dataframe = pd.get_dummies(df)

dataframe_np = dataframe.values

d_len = len(dataframe)
d_test_split = d_len-10
d_columns = dataframe.shape[1]

y = dataframe_np[:,0]
y = y.reshape(d_len,1)
scaler = MinMaxScaler(feature_range=(-1,1))
y= scaler.fit_transform(y)
y= y.reshape(d_len,1)
y = y.reshape(d_len,1,1)
y_test = y[d_test_split:]
y = y[:d_test_split]


x = dataframe_np[:,1:]
sc = MinMaxScaler(feature_range=(-1,1))
#x = sc.fit_transform(x)
x = x.reshape(d_len,1,d_columns-1)
x_test = x[d_test_split:]
x = x[:d_test_split]

# mean = y.mean(axis=0)
# y -= mean
# std = y.std(axis=0)
# y /=std

import random
lis=[]

d = 0.15
l = 1
lr = 0.001
b_size = 3
e = 300
lis.append((d,l,lr,b_size,e))
print(lis)

for (d,l,lr,b_size,e) in lis:
    #initialising a model
    input_layer = Input(name='input_layer',shape=(None,x.shape[-1]))
    
    #hidden layers
    lstm_1 = LSTM(l,name='lstm_1',dropout=d,recurrent_dropout=0, return_sequences=True)(input_layer)
    lstm_2 = LSTM(l,name='lstm_2',dropout=d,recurrent_dropout=0, return_sequences=True)(lstm_1)#, return_sequences=True

    #output layer
    output_layer = Dense(1,name='out_1')(lstm_2)
    model = Model(input_layer, output_layer)
    model.compile(optimizer = Adam(lr=lr), loss='mse')

    fit = model.fit(x, y,batch_size=b_size, epochs=e, verbose=1, validation_split=0.1, shuffle=True)
    loss = fit.history['loss']
    val_loss = fit.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.legend()
    plt.show()
    oo=str.replace(('picPredictLostB3'+','+str(d)+','+str(l)+','+str(lr)+','+str(b_size)+','+str(e)+'BBBBBAAA'),'.','_')
    plt.savefig(oo)
    
    preds = model.predict(x_test, steps=1)
    for i in range(len(x_test)):
        print( "X=%s, Predicted=%s" % (scaler.inverse_transform(y_test[i]),scaler.inverse_transform(preds[i])))
       # print(math.sqrt(preds[i]))
    
    model.save('modelLost3'+','+str(d)+','+str(l)+','+str(lr)+','+str(b_size)+','+str(e)+'BBBBBAAA')
    
def plotInfo():    
    #sns.lmplot(data = dataframe, x='Sales', y='Existing book', hue='Sales', fit_reg=False)
    sns.pairplot(df2, size=2.5)
    plt.tight_layout()
    plt.show()
    plt.savefig("AllPyPlot")
    