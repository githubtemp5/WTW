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
from random import randint
    
__main__ = "__main__"
features = ['Year', 'Month','Existing book']
output = 'Sales'
file_path = "C:\\Users\\ShresthaAl\\Documents\\Datasets\\PMI_dataset_large_2_converted.csv"
models_path = "C:\\Users\\ShresthaAl\\Documents\\models\\"
learning_rate = 0.01
epoch_rate = 100
batch_size_rate = 1

#training split rate between 0 and 1; so 0.2 saves 20% of the data for training
training_split=0.8

#validation split between 0 and 1; so 0.3 saves 30% of the training data for validation
validation_split_rate = 0.1


#np.random.seed(1337)

#sns.lmplot(data = dataframe, x='x1', y='x2', hue='y', fit_reg=False)
#plt.show()    
    
class CustomModel:
    
    def __init__(self):  
          
        #initialising a model
        input_layer = Input(shape =(len(features),))
        
        #hidden layers
        dense_layer_1 = layers.Dense(5, activation='relu')(input_layer)
        dense_layer_2 = layers.Dense(5, activation='relu')(dense_layer_1)
        #dense_layer_3 = layers.Dense(5, activation='relu')(dense_layer_1)
        
        #output layer
        output_layer = layers.Dense(1, activation='linear')(dense_layer_2)
        self.model = Model(input_layer, output_layer)
        self.model.summary()
        
    def train(self):

        stogd = SGD(learning_rate)
        
        #compiling the model, telling keras what to minimise.
        self.model.compile(loss='mse', optimizer=stogd, metrics=['mse'])
        
        #train the model
        self.model.fit(X_train, y_train, batch_size=batch_size_rate, epochs = epoch_rate, validation_split = validation_split_rate, callbacks=[tensorboard])
        
    def evaluateModel(self):
        notRel, X_te, notRel2, y_te = preprocess (X_train, X_test, y_train, y_test, True)
        score = self.model.evaluate(X_te, y_te)
        print('SCORE:    ')
        print(score)
    
    def saveModel(self, modelName):
        self.model.save(models_path+modelName)
        print("Model saved at :", models_path+modelName)
        
    #predict
    def customPredict(self, index):
       # predData = pd.read_csv("C:\\Users\\ShresthaAl\\Documents\\Datasets\\predictValues.csv" , sep=',')
        print('Predictions: ')
        predicts = self.model.predict(X_test.iloc[index:][features], steps=5)
        print(X_test[index:])
        for i in range(0,len(X_test[index:])):
            print(predicts[i][0])
        
    def loadM(self,file_name):
        self.model = load_model(models_path+file_name)

        
#preprocessing data

def preprocess(X_tr, X_te, y_tr, y_te, testProcess):
    
    mean = X_tr.mean(axis=0)
    X_tr -= mean
    std = X_tr.std(axis=0)
    X_tr /= std
    
    mean = y_tr.mean(axis=0)
    y_tr -= mean
    std = y_tr.std(axis=0)
    y_tr /= std
    
    if testProcess:
        mean = X_te.mean(axis=0)
        X_te -= mean
        std = X_te.std(axis=0)
        X_te /= std
        
        
        mean = y_te.mean(axis=0)
        y_te -= mean
        std = y_te.std(axis=0)
        y_te /= std
    
    return X_tr, X_te, y_tr, y_te
    
     #plot scattergraph   
def plotInfo():

    #cols = ['Year', 'Month', 'Existing book', 'New Business', 'Lost Business', 'Sales']
    
    #sns.lmplot(data = dataframe, x='Sales', y='Existing book', hue='Sales', fit_reg=False)
    sns.pairplot(dataframe[dataframe.columns], size=2.5)
    plt.tight_layout()
    plt.show()


if(__main__ == "__main__"):

    dataframe = pd.read_csv(file_path , sep=',')
    

    train_index = dataframe.sample(frac=training_split, random_state=randint(5,250)).index
    #assigning training data
    X_train, y_train = dataframe[features],dataframe[output]
        
    #assigning testing data
    X_test, y_test = dataframe.drop(index=train_index)[features],dataframe.drop(index=train_index)[output]
    
    #initialising tensorboard
    tensorboard = TensorBoard(log_dir="C:\\Users\\ShresthaAl\\Documents\\results\\{}".format(time()))
    
    
    X_train, X_test, y_train, y_test = preprocess(X_train, X_test, y_train, y_test, False)
    m = CustomModel()