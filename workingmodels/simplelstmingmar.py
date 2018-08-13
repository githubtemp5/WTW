"""
LSTM HALF WORKING
"""
from time import time
import os
from random import shuffle

import numpy as np
import pandas as pd

from keras import Input, layers
from keras.models import Sequential, Model
from keras.layers import LSTM
from keras.layers.core import Dropout

# Tell keras to use tensorflow as back-end
os.environ['KERAS_BACKEND'] = 'tensorflow'

import pandas as pd
import seaborn as sns
from matplotlib import pyplot

from keras.callbacks import TensorBoard
from keras.optimizers import SGD, RMSprop

from sklearn.preprocessing import MinMaxScaler

from numpy import array


load_dataset_dir='C:\\Users\\ChoudhuryMB\\Desktop\\keras-work\\using_dataset\\lstm.csv'
load_prediction_dataset_dir='C:\\Users\\ChoudhuryMB\\Desktop\\book1.csv'
save_model_dir='C:\\Users\\ChoudhuryMB\\Desktop\\keras-work\\model\\'
load_model_dir='C:\\Users\\ChoudhuryMB\\Desktop\\keras-work\\model\\'


df = pd.read_csv(load_dataset_dir, header=None, sep=',')
df = df.values


def save_model(model_name):
    model.save(save_model_dir + model_name)
    print("Your model has been saved in:", save_model_dir + model_name)

def preprocess_features(df):
    selected_features = df[:, [0,1]]
    selected_features = selected_features.reshape(len(selected_features), 2)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(selected_features)
    return scaled_values

def preprocess_targets(df):
    output_target = df[:, 2]
    output_target = output_target.reshape(len(output_target), 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(output_target)
    #scaled_values = scaler.fit_transform(output_target)
    return scaled_values


training_examples = preprocess_features(df[0:190])
training_targets = preprocess_targets(df[0:190])
test_examples = preprocess_features(df[190:200])
test_targets = preprocess_targets(df[190:200])

training_examples = training_examples.reshape(len(training_examples), 1, 2)
test_examples=test_examples.reshape(len(test_examples), 1, 2)


print(training_examples.shape)



def make_model():
    model = Sequential()
    model.add(LSTM(18, batch_input_shape=(1, 2), stateful=True))
    model.add(Dropout(0.2))
    model.add(LSTM(12))
    model.add(layers.Dense(1))
    
    model.summary()
    
    return model

print("Creating model")

model = make_model()


print("Compiling model")
model.compile(optimizer='Adam',
             loss='mse',
             metrics=['mae'])

tensorboard = TensorBoard(log_dir='C:\\Users\\ChoudhuryMB\\Desktop\\keras-work\\test1\\{}'.format(time()))

print("Starting to train...",
      "(Check TensorBoard for Training Loss and Validation loss)")
summary = model.fit(
        training_examples, training_targets,
        epochs=10,
        verbose=1)


score = model.evaluate(test_examples, test_targets, verbose=1)

print('Test loss:', (score[0]))
print('Test accuracy:', (score[1]))

 

forecasts = model.predict(test_examples, batch_size=10)


def covert_prediction(forecasts):
    predictions = forecasts
    converted_predictions = list()
    
    for prediction in range((len(predictions))):
        predictions = predictions[prediction]
        predictions = predictions.reshape(1, len(predictions))
        scaler = MinMaxScaler(feature_range=(-1, 1))
        #invert_scale = scaler.fit_transform(predictions)
        invert_scale = scaler.inverse_transform(predictions)
        invert_scale = invert_scale[0, :]
        converted_predictions.append(invert_scale)
    
    return converted_predictions
        

covert_prediction(forecasts)

"""
def inverse_transform(series, forecasts, scaler, n_test):
	inverted = list()
	for i in range(len(forecasts)):
		# create array from forecast
		forecast = array(forecasts[i])
		forecast = forecast.reshape(1, len(forecast))
		# invert scaling
		inv_scale = scaler.inverse_transform(forecast)
		inv_scale = inv_scale[0, :]
		# invert differencing
		index = len(series) - n_test + i - 1
		last_ob = series.values[index]
		inv_diff = inverse_difference(last_ob, inv_scale)
		# store
		inverted.append(inv_scale)
	return inverted
"""
def make_f():
    pass
