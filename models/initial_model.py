"""
SIMPLE MACHINE LEARNING FOR REVENUE PREDICTION
BASED ON PREVIOUS YEARS
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime


df = pd.read_csv('C:\\Users\\ChoudhuryMB\\Documents\\ds\\modpmi.csv', sep=',')

df = df.reindex(np.random.permutation(df.index))

def preprocess_features(df):
    
    selected_features = df[["Base_Amount", "Transaction_Date"]]
    return selected_features

def preprocess_targets(df):
    
    output_targets = pd.DataFrame()
    output_targets = df["Base_Amount"]
    return output_targets

# Choose the first 143710 (out of 179638) ~ 80% Available data for training
training_examples = preprocess_features(df.head(500))
training_targets = preprocess_targets(df.head(500))

# Choose the first 35928 (out of 179638) ~ 20% Available data for validation
validation_examples = preprocess_features(df.tail(500))
validation_targets = preprocess_targets(df.tail(500))

#Double-check that we've done the right thing.
print("#---------------------------#")
print("SAMPLE PARTITION:")
print("Training examples summary:")
print(training_examples.describe())
print(" ")
print("Validation examples summary:")
print(validation_examples.describe())
print(" ")
print("Training targets summary:")
print(training_targets.describe())
print(" ")
print("Validation targets summary:")
print(validation_targets.describe())
print("#---------------------------#")
      

#If we have empty data set we change it to -99999
df.fillna(value=-99999, inplace=True)
df.dropna(inplace=True)

#Definining new column for forecast
df['label'] = df['Base_Amount']

# Defining X and Y values for ML
X = np.array(df.drop(['label'], 1))
