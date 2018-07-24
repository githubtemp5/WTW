import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

__main__ = "__main__"

#sns.lmplot(data = dataframe, x='x1', y='x2', hue='y', fit_reg=False)
#plt.show()

#preprocessing data
def preprocess(X_train, X_test):
    mean = X_train.mean(axis=0)
    X_train -= mean
    std = X_train.std(axis=0)
    X_train /= std
    
    mean = X_test.mean(axis=0)
    X_test -= mean
    std = X_test.std(axis=0)
    X_test /= std
    
    return X_train, X_test
    
def initialiseModel():
    
    #initialising a model
    model = Sequential('Regression_Model')
    model.add(layers.Dense(name="Input_Layer", units=3, input_dim=2))
    model.add(layers.Activation(name='rel', activation='relu'))
    model.summary()
    
    #stochastic gradient descent, learning rate
    stogd = SGD(0.01)
    
    #compiling the model, telling keras what to minimise.
    model.compile(loss='mse', optimizer=stogd, metrics=['mse'])

#train the model
#model.fit(X_train, y_train, epochs=100, batch_size =1)

#prediction = model.predict(dataframe.values)

if(__main__ == "__main__"):

    dataframe = pd.read_csv("moons.csv", sep=',')

    #assigning training data
    X_train, y_train = dataframe.iloc[train_index][['x1','x2']],dataframe.iloc[train_index]['y']
        
    #assigning testing data
    X_test, y_test = dataframe.drop(index=train_index)[['x1','x2']],dataframe.drop(index=train_index)['y']
    
    #initialising tensorboard
    tensorboard = TensorBoard(log_dir="C:\\Users\\ShresthaAl\\Documents\\results")
    
    print(X_train)
    X_train, X_test = preprocess(X_train, X_test)
    print(X_train)
   #initialiseModel()