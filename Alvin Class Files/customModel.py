#imports
from time import time
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense
from keras.callbacks import TensorBoard
from keras.optimizers import SGD
from keras import Input, layers
    
__main__ = "__main__"
features = ['Year', 'Month', 'Existing book']
output = 'Sales'
file_path = "C:\\Users\\ShresthaAl\\Documents\\Datasets\\PMI_dataset_large_2_converted.csv"
learning_rate = 0.01
#np.random.seed(1337)

#sns.lmplot(data = dataframe, x='x1', y='x2', hue='y', fit_reg=False)
#plt.show()    
    
class CustomModel:
    
    def __init__(self):  
          
        #initialising a model
        input_layer = Input(shape =(len(features),))
        dense_layer_1 = layers.Dense(3, activation='relu')(input_layer)
        output_layer = layers.Dense(1)(dense_layer_1)
        
        self.model = Model(input_layer, output_layer)
        self.model.summary()
        
        
        stogd = SGD(learning_rate)
        
        #compiling the model, telling keras what to minimise.
        self.model.compile(loss='mse', optimizer=stogd, metrics=['mse'])
        

        #train the model
        self.model.fit(X_train, y_train, batch_size=10, epochs = 1000, validation_split = 0.1, callbacks=[tensorboard])
        
        score = self.model.evaluate(X_train, y_train)
        print(score)
        # input_tensor = Input(shape =(64,))
        # x = layers.Dense(32, activation='relu')(input_tensor)
        # x = layers.Dense(32, activation='relu')(x)
        # output_tensor = layers.Dense(10, activation='softmax')(x)
        # model = Model(input_tensor, output_tensor)
        # model.summary()
        
        
        
        
    #predict
    def customPredict(self,a):
        predData = pd.read_csv("C:\\Users\\ShresthaAl\\Documents\\Datasets\\predictValues.csv" , sep=',')
        print('Predictions: ')
        predicts = self.model.predict(predData.iloc[a:a+1][features])
        print(predicts)
        
#preprocessing data

def preprocess(X_train, X_test, y_train, y_test):
    
    mean = X_train.mean(axis=0)
    X_train -= mean
    std = X_train.std(axis=0)
    X_train /= std
    
    mean = y_train.mean(axis=0)
    y_train -= mean
    std = y_train.std(axis=0)
    y_train /= std
    
    print(X_train)
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
    
     #plot scattergraph   
def plotInfo():

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
    
    
    X_train, X_test, y_train, y_test = preprocess(X_train, X_test, y_train, y_test)
    m = CustomModel()