#imports
from time import time
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, SimpleRNN, Embedding
from keras.callbacks import TensorBoard
from keras.optimizers import SGD
from keras import Input, layers
from random import randint
from sklearn.model_selection import train_test_split

__main__ = "__main__"
features = ['Year', 'Month']
output = ['Sales']
file_path = "C:\\Users\\ShresthaAl\\Documents\\Datasets\\PMI_dataset_large_updated_lossBusiness.csv"
models_path = "C:\\Users\\ShresthaAl\\Documents\\models\\"
learning_rate = 0.015
epoch_rate = 70
batch_size_rate = 10


timesteps = 100
input_features=32
output_features=64

inputs = np.random.random((timesteps, input_features))

state_t = np.zeros((output_features,))

W = np.random.random((output_features, input_features))

U = np.random.random((output_features, output_features))
b = np.random.random((output_features,))

successive_outputs = []
state_t=0
for input_t in inputs:
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    successive_outputs.append(output_t)
    state_t = output_t

final_output_sentence = np.concatenate(successive_outputs, axis=0)
