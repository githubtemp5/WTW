#imports
from time import time
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
from keras.optimizers import SGD
    
__main__ = "__main__"
features = ['Year', 'Month', 'Existing book']
output = 'Sales'
file_path = "C:\\Users\\ShresthaAl\\Documents\\Datasets\\PMI_dataset_large_2_converted.csv"
learning_rate = 0.01

#sns.lmplot(data = dataframe, x='x1', y='x2', hue='y', fit_reg=False)
#plt.show()

class Model:

    #preprocessing data
    def preprocess(self,X_train, X_test, y_train, y_test):
        mean = X_train.mean(axis=0)
        X_train -= mean
        std = X_train.std(axis=0)
        X_train /= std
        
        mean = y_train.mean(axis=0)
        y_train -= mean
        std = y_train.std(axis=0)
        y_train /= std
        
        # mean = X_test.mean(axis=0)
        # X_test -= mean
        # std = X_test.std(axis=0)
        # X_test /= std
        
        # 
        # mean = y_test.mean(axis=0)
        # y_test -= mean
        # std = y_test.std(axis=0)
        # y_test /= std
        
        return X_train, X_test, y_train, y_test
        
    def initialiseModel(self):
        
        #initialising a model
        self.model = Sequential(name='Regression_Model')
        
        #input layer
        self.model.add(Dense(name="Layer1", units=3, input_dim=len(features), activation = 'relu'))
        self.model.summary()
        
        #output layer
        self.model.add(Dense(name='OutputLater', units=1))
        #model.add(layers.Activation(name='sig', activation='sigmoid'))
        
        #stochastic gradient descent, learning rate decay
        stogd = SGD(learning_rate)
        
        #compiling the model, telling keras what to minimise.
        self.model.compile(loss='mse', optimizer=stogd, metrics=['mse'])
    
        #train the model
        self.model.fit(X_train, y_train, batch_size=10, epochs = 1200, validation_split = 0.2, callbacks=[tensorboard])
        
    #predict
    def predict(self, a, b):
        predData = pd.read_csv("C:\\Users\\ShresthaAl\\Documents\\Datasets\\predictValues.csv" , sep=',')
        print('Predictions: ')
        predicts = self.model.predict(predData.iloc[a:b][features], verbose=0, steps=10)
        print(predicts[:][0])
def plotInfo():
    #plot scattergraph
    
    cols = ['Year', 'Month', 'Existing book', 'New Business', 'Lost Business', 'Sales']
    
    #sns.lmplot(data = dataframe, x='Sales', y='Existing book', hue='Sales', fit_reg=False)
    sns.pairplot(dataframe[cols], size=2.5)
    plt.tight_layout()
    plt.show()
    
    

if(__main__ == "__main__"):

    dataframe = pd.read_csv(file_path , sep=',')
    

    train_index = dataframe.sample(frac=0.6, random_state=21).index
    #assigning training data
    X_train, y_train = dataframe.iloc[train_index][features],dataframe.iloc[train_index][output]
        
    #assigning testing data
    X_test, y_test = dataframe.drop(index=train_index)[features],dataframe.drop(index=train_index)[output]
    
    #initialising tensorboard
    tensorboard = TensorBoard(log_dir="C:\\Users\\ShresthaAl\\Documents\\results\\{}".format(time()))
    
    m = Model()
    print(X_train)
    X_train, X_test, y_train, y_test = m.preprocess(X_train, X_test, y_train, y_test)
    print(X_train)
    m.initialiseModel()