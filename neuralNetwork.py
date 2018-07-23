import pandas as pd
from keras.models import Sequential
import seaborn as sns
from keras.callbacks import TensorBoard

from keras import layers
[layer for layer in dir(layers) if not layer.startswith('_')]

from keras import optimizers
[opt for opt in dir(optimizers) if not opt.startswith('_')]

__main__="__main__"
    
class Model:
    
    def __init__(self, modelName):
        self.model = Sequential(name = modelName)
        self.initialise()
        
    def initialise(self):
        
    
    # input layer
        self.model.add(layers.Dense(name='InputLayer',units = 3, input_dim=2))
        self.model.add(layers.BatchNormalization())
    
    #hidden layer 1
        self.model.add(layers.Dense(name='FullyConnected_1', units=2))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Activation(name='sigmoid1', activation='sigmoid'))
        self.model.add(layers.Dropout(0.3))
        
    #hidden layer 2
        # self.model.add(layers.Dense(name='HL_3', units=5))
        # self.model.add(layers.BatchNormalization())
        # self.model.add(layers.Activation(name='sigmoid3', activation='sigmoid'))
        # self.model.add(layers.Dropout(0.3))
    
    #output layer
        self.model.add(layers.Dense(name='FullyConnected2_OutputLater', units=1))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Activation(name='sigmoid2', activation='sigmoid'))
        
        self.model.summary()
        self.model.compile(
                optimizer='Adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
        print('Model config: ')
        self.model.get_config()
        
    def train(self):
        self.summary = self.model.fit(X_train, y_train, batch_size=4, epochs = 100, validation_split = 0.1, verbose=0,callbacks=[tensorboard])
        self.score = self.model.evaluate(X_test, y_test, verbose=0)
        print('Test loss: ', self.score[0])
        print('Test Accuracy: ', self.score[1])
            
if __main__ =="__main__":
    #initialising file reading
    dataframe = pd.read_csv("C:\\Users\\ShresthaAl\\Documents\\moons.csv", sep=',')
    print('Rows x Columns', dataframe.shape)
    
    #sns.lmplot(data = dataframe, x='Year', y='Existing book', hue='Year', fit_reg=False)
    
    train_index = dataframe.sample(frac=0.6, random_state=21).index
    
    #assigning training data
    X_train, y_train = dataframe.iloc[train_index][['x1','x2']], dataframe.iloc[train_index]['y']
    
    #assigning testing data
    X_test, y_test = dataframe.drop(index=train_index)[['x1','x2']],dataframe.drop(index=train_index)['y']
    
    tensorboard = TensorBoard(log_dir="C:\\Users\\ShresthaAl\\Documents\\results")
    
    #creating an instance of a neural network
    m = Model('Real Data')
    m.train()