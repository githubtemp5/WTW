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

df = pd.read_csv("C:\\Users\\ShresthaAl\\Downloads\\Appliances-energy-prediction-data-master\\Appliances-energy-prediction-data-master\\training.csv", dtype={'WeekStatus':str, 'Day_of_week':str})#'Year':str, 'Month':str}

#df2 = pd.read_csv("C:\\Users\\ShresthaAl\\Documents\\Datasets\\PMI_dataset_large_updated_lossBusiness3.csv")
print(df.describe())

#preprocessing data
#converting dates to 0 and 1 bag of words technique?
dataframe = pd.get_dummies(df)

#dataframe to numpy array
dataframe_np = dataframe.values

d_len = len(dataframe)
d_test_split = d_len-14003
d_columns = dataframe.shape[1]

#scale and assign test and training data for target(y)
y = dataframe_np[:,0]
y = y.reshape(d_len,1)
scaler = MinMaxScaler(feature_range=(-1,1))
y= scaler.fit_transform(y)
y= y.reshape(d_len,1)
y = y.reshape(d_len,1,1)
y_test = y[d_test_split:810]
y = y[:d_test_split]


#scale and assign test and training data for features(x)
x = dataframe_np[:,1:]
sc = MinMaxScaler(feature_range=(-1,1))
#x = sc.fit_transform(x)
x = x.reshape(d_len,1,d_columns-1)
x_test = x[d_test_split:810]
x = x[:d_test_split]

# mean = y.mean(axis=0)
# y -= mean
# std = y.std(axis=0)
# y /=std

import random
lis=[]

drop_rate = 0.1
neuron_units = 256
l_rate = 0.001
b_size = 10
ep = 20
lis.append((drop_rate,neuron_units,l_rate,b_size,ep))
print(lis)

for (drop_rate,neuron_units,l_rate,b_size,ep) in lis:
    #initialising a model
    input_layer = Input(name='input_layer', shape=(None,x.shape[-1]), batch_shape=(b_size,1,x.shape[2]))
    
    #hidden layers
    lstm_1 = LSTM(neuron_units,name='lstm_1',dropout=drop_rate,recurrent_dropout=0, stateful=True, return_sequences=True)(input_layer)
    lstm_2 = LSTM(neuron_units,name='lstm_2',dropout=drop_rate,recurrent_dropout=0,return_sequences=True, stateful=True)(lstm_1)#

    #output layer
    output_layer = Dense(1,name='out_1')(lstm_2)
    model = Model(input_layer, output_layer)
    model.compile(optimizer = Adam(lr=l_rate), loss='mse')

    #training the model
    fit = model.fit(x, y, batch_size=b_size, epochs=ep, verbose=1, validation_split=0.1, shuffle=True)
    
    #plotting the validation and loss graph
    loss = fit.history['loss']
    val_loss = fit.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.legend()
    plt.show()
    
    oo=str.replace(('picDateSales5'+';'+str(drop_rate)+';'+str(neuron_units)+';'+str(l_rate)+';'+str(b_size)+';'+str(ep)),'.','_')
    plt.savefig(oo)
    
    preds = model.predict(x_test, batch_size=b_size )#steps=1
    for i in range(len(x_test)):
        print( "X=%s, Predicted=%s" % (scaler.inverse_transform(y_test[i]),scaler.inverse_transform(preds[i])))
       # print(math.sqrt(preds[i]))
    
    model.save('modelTemp6'+';'+str(drop_rate)+';'+str(neuron_units)+';'+str(l_rate)+';'+str(b_size)+';'+str(ep))
    
def plotInfo():    
    #sns.lmplot(data = dataframe, x='Sales', y='Existing book', hue='Sales', fit_reg=False)
    sns.pairplot(df2, size=2.5)
    plt.tight_layout()
    plt.show()
    plt.savefig("AllPyPlot")
    