"""
SIMPLE MACHINE LEARNING FOR REVENUE PREDICTION
BASED ON PREVIOUS YEARS
"""

from __future__ import print_function

import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset


# establish the access to the data and convert to dataframe
df = pd.read_csv('C:\\Users\ChoudhuryMB\\Documents\\ds\\converted-data.csv',
                     header=0,
                     encoding = "ISO-8859-1",
                     sep=',',
                     error_bad_lines=False,
                     index_col=0,
                     dtype='unicode')

# makes sure the data is randomized to avoid overfitting
df = df.reindex(np.random.permutation(df.index))

def preprocess_features(df):
    """
    FEATURES SELECTION
    Args:
        The broker dataframe (called 'df')
    Returns:
        A Dataframe (df) containing the features to be used in the model
    """
    
    selected_features = df[["Base Amount", "Transaction Date"]]
    return selected_features

def preprocess_targets(df):
    """
    LABEL SELECTION
    Args:
        The broker dataframe (called 'df')
    Returns:
        A Dataframe (df) containing the features to be used in the model
    """
    
    output_targets = pd.DataFrame()
    output_targets = df["Base Amount"]
    return output_targets

# Choose the first 143710 (out of 179638) ~ 80% Available data for training
training_examples = preprocess_features(df.head(143710))
training_targets = preprocess_targets(df.head(143710))

# Choose the first 35928 (out of 179638) ~ 20% Available data for validation
validation_examples = preprocess_features(df.tail(35928))
validation_targets = preprocess_targets(df.tail(35928))

#Double-check that we've done the right thing.
#print("Training examples summary:")
#print(training_examples.describe())
#print("Validation examples summary:")
#print(validation_examples.describe())
#print("Training targets summary:")
#print(training_targets.describe())
#print("Validation targets summary:")
#print(validation_targets.describe())



# BUILDING DEEP REGRESSOR NEURAL NETWORK (DNNRegressir)

def construct_feature_columns(input_features):
    """
    Contruct the tensorflow feature columns
    
    Args:
        input features: The names of the numerical input features to use.
    Returns:
        A set of feature columns
    """
    return set([tf.feature_column.numeric_column(my_feature)
                for my_feature in input_features])

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """
    Trains a neural net regression model.
    Args:
        features: pandas DataFrame of features
        targets: pandas DataFrame of targets
        batch_size: Size of batches to be passed to the model
        shuffle: True or False. Whether to shuffle the data.
        num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
        Tuple of (features, labels) for next data batch
    """
    
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}
    
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(10000)
      
     # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def train_nn_regression_model(
        learning_rate,
        steps,
        batch_size,
        hidden_units,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    """
    REGRESSION MODEL NEURAL NETWORK
    
    In addition to training, this function also prints training progress information,
    as well as a plot of the training and validation loss over time.
    
    Args:
        learning_rate:
        steps: 
        batch_size:
        hidden_units:
        training_examples:
        training_targets:
        validation_examples:
        validation_targets:
    Returns:
        A `DNNRegressor` object trained on the training data.
    """
    
    periods = 10
    steps_per_period = steps / periods
    
    #Create DNNRegressor Object
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    dnn_regressor = tf.estimator.DNNRegressor(
            feature_columns=construct_feature_columns(training_examples),
            hidden_units=hidden_units,
            optimizer=my_optimizer)
    
    #Create input functions.
    training_input_fn = lambda: my_input_fn(training_examples,
                                            training_targets["Base Amount"],
                                            batch_size=batch_size)
    
    predict_training_input_fn = lambda: my_input_fn(training_examples, 
                                                    training_targets["Base Amount"], 
                                                    num_epochs=1, 
                                                    shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                      validation_targets["Base Amount"], 
                                                      num_epochs=1, 
                                                      shuffle=False)
    
    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("RMSE (on training data)")
    training_rmse = []
    validation_rmse = []
    
    for period in range (0, periods):
        # Train the model, starting from the prior state.
        dnn_regressor.train(
               input_fn=training_input_fn,
               steps=steps_per_period)
    
        # Take a break and compute predictions.
        training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])
        
        validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
        
        # Compute training and validation loss.
        training_root_mean_squared_error = math.sqrt(
                metrics.mean_squared_error(training_predictions, training_targets))
        validation_root_mean_squared_error = math.sqrt(
                metrics.mean_squared_error(validation_predictions, validation_targets))
        
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
        
        # Add the loss metrics from this period to our list.
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
    print("Model training finished.")
    
    # Output a graph of loss metrics over periods.
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()

    print("Final RMSE (on training data):   %0.2f" % training_root_mean_squared_error)
    print("Final RMSE (on validation data): %0.2f" % validation_root_mean_squared_error)
    
    return dnn_regressor

print("Starting the DNN Regresor model")

dnn_regressor = train_nn_regression_model(
        learning_rate=0.01,
        steps=500,
        batch_size=10,
        hidden_units=[10, 2],
        training_examples=training_examples,
        training_targets=training_targets,
        validation_examples=validation_examples,
        validation_targets=validation_targets)

df = pd.read_csv('C:\\Users\ChoudhuryMB\\Documents\\ds\\converted-data.csv',
                     header=0,
                     encoding = "ISO-8859-1",
                     sep=',',
                     error_bad_lines=False,
                     index_col=0,
                     dtype='unicode')

test_examples = preprocess_features(df)
test_targets = preprocess_targets(df)

predict_testing_input_fn = lambda: my_input_fn(test_examples,
                                               test_targets["Base Amount"],
                                               num_epochs=1,
                                               shuffle=False)

test_predictions = dnn_regressor.predict(input_fn=predict_testing_input_fn)
test_predictions = np.array([item['predictions'][0] for item in test_predictions])

root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(test_predictions, test_targets))
print("Final RMSE (on test data): %0.2f" % root_mean_squared_error)
