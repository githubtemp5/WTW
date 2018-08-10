"""
@author         :Alvin Shrestha
@Organisation   :Willis Towers Watson
@contact        :githubtemp@gmail.com
@link           :github.com/githubtemp5
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import math
import keras
import time
from PIL import Image
from keras.models import load_model, Model
from keras.layers import Dense, LSTM, Input
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
__main__='__main__'

datasets_dir = "C:\\Users\\ShresthaAl\\Documents\\Datasets\\"               #folder where dataset(s) is stored
model_save_dir = "C:\\Users\\ShresthaAl\\Documents\\models2\\"              #folder where generated model files are to be saved
model_graph_dir = "C:\\Users\\ShresthaAl\\Documents\\graphs\\"              #folder where generated graphs are to be saved
model_config_dir = "C:\\Users\\ShresthaAl\\Documents\\model_config\\"       #folder where information about the model is to be saved
main_file = "PMI_dataset_large_updated_lossBusiness.csv"                    #main file for training and testing

batchSize = 2
neuronUnits = 30
learningRate =0.001
noOfEpochs = 200
validationSplit = 0.4
dropoutRate = 0.2
recurrentDropoutRate = 0.0

class CustomModel:
    def __init__(self,b_size): 
        """
        @constructor
        b_size          : Batch size to process in each epoch
        """
        
        self.b_size = b_size
    
    def load_excel_file(self):
        """
        @method
            This method loads the data from the csv/excel file and stores it in the self.data variable.
            It also prints the description of the data.
            self.data    : Stores the loaded data as a dataframe format.
        """
        #this specifies which of the data is in string condition and to be read as string in the program.
        self.data = pd.read_csv(datasets_dir + main_file)#'Year':str,
        print(self.data.describe())
        
        print("Files successfully loaded")
    
    def preprocess_data(self):
        """
        @method
        This method prepares the raw data for training.
            
            self.data_np    : Converts the self.data DataFrame into a Numpy format.
            self.d_len      : Number of rows in the loaded data.
            self.split_row  : The row of separation of training and testing data; It is total number of rows - specified batch size
            self.y          : The target/output/labels for the training data
            self.scaler     : It normalises/scales the data between the specified parameters, For eg: (-1,1):
                                Lowest number is -1 and the highest number is 1
            self.y_test     : The target/output/lables for the testing data
            self.x          : The features/input of the testing data
            self.x_test     : The target/output/labels for the testing data
        """
        print("Starting Preprocess....")
        #preprocessing data
        
        #TRAINING preprocess
        #converting string data to 1's and 0's
        #self.data = pd.get_dummies(self.data)
        #dataframe to numpy array
        self.data_np = self.data.values
        
        self.d_len = len(self.data_np)
        self.d_columns = self.data_np.shape[1]
        self.split_row = self.d_len-self.b_size
        
        #scale and assign training data for target(y) and features(x)
        #for targets
        self.y = self.data_np[:,2]
        self.y = self.y.reshape(self.d_len,1)
        self.scaler = MinMaxScaler(feature_range=(-1,1))
        self.y = self.scaler.fit_transform(self.y)
        self.y = self.y.reshape(self.d_len,1,1)
        
        self.y_test = self.y[self.split_row:self.d_len]
        self.y = self.y[:self.split_row]
        
        #for features
        self.x = self.data_np[:,0:2]
        self.x = self.x.reshape(self.d_len,self.x.shape[1])
        self.scaler2 = MinMaxScaler(feature_range=(-1,1))
        self.x = self.scaler2.fit_transform(self.x)
        self.x = self.x.reshape(self.d_len,1,self.d_columns-1)
        self.x_test = self.x[self.split_row:self.d_len]
        self.x = self.x[:self.split_row]
        
        print("Data preprocessed successfully!")
    
    def define_model(self,dropout_rate, rcurr_dropout, neuron_units,learning_rate,epochs):
        """
        @method
        Values to pass when defining a model:
        
        dropout_rate    : Dropout rate for the LSTM layer
        rcurr_dropout   : Recurrent dropout rate
        neuron_units    : Number of units used in all the hidden layers
        learning_rate   : Learning rate for the model
        epochs          : Number times the model should train
        
        Uses keras functional API
        
        input_layer     : Input layer for the network; name = name for layer; batch_shape = (batch_size, 1, no_of_columns_in_features i.e x)
        lstm_1          : LSTM First hidden layer
        lstm_2          : LSTM Second hidden layer
        output_layer    : Output Dense layer
        self.model      : This variable stores the compiled model
                            
        """
        self.dropout_rate = dropout_rate
        self.rcurr_dropout = rcurr_dropout
        self.neuron_units = neuron_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        #initialising the input layer
        input_layer = Input(name='input_layer', batch_shape=(self.b_size,1,self.x.shape[-1]))
        
        #hidden layers
        lstm_1 = LSTM(neuron_units,
                      name='lstm_1',
                      dropout=dropout_rate,
                      return_state = True,
                      recurrent_dropout=rcurr_dropout,
                      return_sequences=True)(input_layer)
                      
        lstm_2 = LSTM(neuron_units,
                      name='lstm_2',
                      dropout=dropout_rate,
                      return_sequences=True,
                      recurrent_dropout=rcurr_dropout)(lstm_1)
        
        #output layer
        output_layer = Dense(1,name='output_layer')(lstm_2)
        
        #compiling model
        self.model = Model(input_layer, output_layer)
        self.model.compile(optimizer = Adam(lr=learning_rate), loss='mse')
        
        print('Model successfully compiled')
        print('Model Summary:')
        print(self.model.summary())
        
    def load_saved_model(self, filepath):
        """
        @method
            Loads an existing model from a given filepath to the system
            It also shows the loss, validation image for the model
        """
        self.dropout_rate = dropoutRate
        self.rcurr_dropout = recurrentDropoutRate
        self.neuron_units = neuronUnits
        self.learning_rate = learningRate
        self.epochs = noOfEpochs
        
        self.model = load_model(model_save_dir+filepath)
        print('Model Loaded')
        
        f = Image.open(model_graph_dir+filepath+".png")
        f.show()
    
    def train_model(self,validation_split):
        """
        @method
        This method trains the current model
            
            self.validation_split    = Validation split between a value of 0 and 1
                                        Example: 0.2 is 20% of the training data is set aside for validation
            model_fit_history        = Stores an history object which has information about the training(fitting) of the model
        """
        
        self.validation_split = validation_split
        #training the model
        model_fit_history = self.model.fit(self.x,
                                           self.y,
                                           batch_size = self.b_size,
                                           epochs=self.epochs,
                                           verbose=2,
                                           validation_split=self.validation_split,
                                           shuffle=True)
        
        print('Training complete')
        self.display_graph()
        
    def predict_model(self):
        self.preds = self.model.predict(self.x_test,
                                        batch_size=self.b_size)
        self.predictions=[]
        for i in range(len(self.y_test)):
            self.predictions.append((((float)(self.scaler.inverse_transform(self.y_test[i]))),
                                     ((float)(self.scaler.inverse_transform(self.preds[i])))))
        return self.predictions

    def display_graph(self):
        #plotting the validation and loss graph
        loss = self.model.history.history['loss']
        val_loss = self.model.history.history['val_loss']
        plot_epochs = range(1, len(loss) + 1)
        plt.figure()
        plt.plot(plot_epochs, loss, 'bo', label='Training loss')
        plt.plot(plot_epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and Validation loss')
        plt.legend()
        plt.show()
        
        #saves the loss and validation graph
        current_time = time.strftime("%d_%m_%y__%H_%M_%S")
        plt.savefig(model_graph_dir+ current_time)
        
        #saves the model file
        self.save_model(current_time)
        print("Model saved as: ",current_time)
        
        #writes the model configuration into a txt file
        txt_file = open(model_config_dir+current_time+"config.txt", "w+")
        txt_file.write(str(self.model.history.params)+"\n\n"+
                       str(self.model.to_json())+"\n\n"+"Predictions: \n"+
                       str(self.predict_model()))
        txt_file.close()
        

    def save_model(self,current_time):
        """
        @method
        This method saves the current model into the specified directory.
        You should not have to call this method as the model is saved when the training is complete.
        However, if you want to save your model with a different name, you can do so by
        passing the command system.save_model("your_file_name")
            """
        self.model.save(model_save_dir+current_time)
        print('Model saved successfully')
        
    def print_model_summary(self):
        print(self.model.summary())
    
    def calculate_val_split(self,val):
        x = self.x.shape[0]
    
if __main__=='__main__':
    """starting point for the program"""
    #specify your bath size
    system = CustomModel(batchSize)
    #loads the excel file onto the system
    system.load_excel_file()
    #preprocesses the raw_data
    system.preprocess_data()
    system.define_model(dropoutRate, recurrentDropoutRate, neuronUnits, learningRate, noOfEpochs)
    #specify your validation split
    system.train_model(validationSplit)