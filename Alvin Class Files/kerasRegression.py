import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.optimizers import SGD

__main__ = "__main__"
features = ['Existing book', 'New Business']
output = 'Sales'
file_path = "PMI_dataset_large.csv"

#sns.lmplot(data = dataframe, x='x1', y='x2', hue='y', fit_reg=False)
#plt.show()



#preprocessing data
def preprocess(X_train, X_test, y_train, y_test):
    mean = X_train.mean(axis=0)
    X_train -= mean
    std = X_train.std(axis=0)
    X_train /= std
    
    mean = X_test.mean(axis=0)
    X_test -= mean
    std = X_test.std(axis=0)
    X_test /= std
    
    mean = y_train.mean(axis=0)
    y_train -= mean
    std = y_train.std(axis=0)
    y_train /= std
    
    mean = y_test.mean(axis=0)
    y_test -= mean
    std = y_test.std(axis=0)
    y_test /= std
    
    return X_train, X_test, y_train, y_test
    
def initialiseModel():
    
    #initialising a model
    model = Sequential(name='Regression_Model')
    
    #input layer
    model.add(layers.Dense(name="Layer1", units=3, input_dim=len(features), activation = 'relu'))
    model.summary()
    
    #output layer
    model.add(layers.Dense(name='OutputLater', units=1))
    model.add(layers.Activation(name='sig', activation='sigmoid'))
    
    #stochastic gradient descent, learning rate decay
    stogd = SGD(0.01)
    
    #compiling the model, telling keras what to minimise.
    model.compile(loss='mse', optimizer=stogd, metrics=['mse'])

    #train the model
    model.fit(X_train, y_train, batch_size=10, epochs = 100, validation_split = 0.1)
    

if(__main__ == "__main__"):

    dataframe = pd.read_csv(file_path , sep=',')
    
    train_index = dataframe.sample(frac=0.6, random_state=21).index
    #assigning training data
    X_train, y_train = dataframe.iloc[train_index][features],dataframe.iloc[train_index][output]
        
    #assigning testing data
    X_test, y_test,  = dataframe.drop(index=train_index)[features],dataframe.drop(index=train_index)[output]
    
    #initialising tensorboard
    tensorboard = TensorBoard(log_dir="C:\\Users\\ShresthaAl\\Documents\\results")
    
    print(X_train)
    X_train, X_test, y_train, y_test = preprocess(X_train, X_test, y_train, y_test)
    print(X_train)
    initialiseModel()