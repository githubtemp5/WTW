"""
@author         :Muklek Bokth Choudhury
@Organisation   :Willis Towers Watson
@link           :www.muklek.com
"""

from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.core import Dropout
from keras.optimizers import Adam
from math import sqrt
from numpy import array
import matplotlib.pyplot as plt
import random


# date-time parsing function for loading the dataset
def parser(x):
	return datetime.strptime(x, '%Y-%m-%d')

 
# convert time series into supervised learning problem
def window_method(data, lookBack=1, delay=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(lookBack, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, delay):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, lookBack, delay):
	# extract raw values
	raw_values = series.values
	raw_values = raw_values.reshape(len(raw_values), 1)
	# rescale values to -1, 1
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaled_values = scaler.fit_transform(raw_values)
	scaled_values = scaled_values.reshape(len(scaled_values), 1)
	# transform into supervised learning problem X, y
	supervised = window_method(scaled_values, lookBack, delay)
	supervised_values = supervised.values
	# split into train and test sets
	train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
	return scaler, train, test
 
# fit an LSTM network to training data
def fit_lstm(neurons, train, lookBack, delay, n_batch, n_epochs, learning_rate):
	# reshape training into [samples, timesteps, features]
	X, y = train[:, 0:lookBack], train[:, lookBack:]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	# design network
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(y.shape[1]))
	model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate))
	fit = model.fit(X, y, epochs=n_epochs, batch_size=n_batch, verbose=1, shuffle=False)

	loss = fit.history['loss']
	epochs = range(1, n_epochs+1)
	plt.figure()
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.title('Training loss')
	plt.legend()
	plt.show()

	return model
 
# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
	# reshape input pattern to [samples, timesteps, features]
	X = X.reshape(1, 1, len(X))
	# make forecast
	forecast = model.predict(X, batch_size=n_batch)
	# convert to array
	return [x for x in forecast[0, :]]
 
# evaluate the persistence model
def make_forecasts(model, n_batch, train, test, lookBack, delay):
	forecasts = list()
	for i in range(len(test)):
		X, y = test[i, 0:lookBack], test[i, lookBack:]
		# make forecast
		forecast = forecast_lstm(model, X, n_batch)
		# store the forecast
		forecasts.append(forecast)
	return forecasts
 
# inverse data transform on forecasts
def inverse_transform(series, forecasts, scaler, n_test):
	inverted = list()
	for i in range(len(forecasts)):
		# create array from forecast
		forecast = array(forecasts[i])
		forecast = forecast.reshape(1, len(forecast))
		# invert scaling
		inv_scale = scaler.inverse_transform(forecast)
		inv_scale = inv_scale[0, :]
		# store
		inverted.append(inv_scale)
	return inverted
 
# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, lookBack, delay):
	for i in range(delay):
		actual = [row[i] for row in test]
		predicted = [forecast[i] for forecast in forecasts]
		rmse = sqrt(mean_squared_error(actual, predicted))
		print('t+%d RMSE: %f' % ((i+1), rmse))

def print_actual_and_forecasts(n_test, n_seq, forecasts, actual):
    for x in range(n_test):
        for i in range(n_seq):
            print("forecasted:", forecasts[x][i], "actual:", actual[x][i])


def model_save(save_model_directory, model, lookBack, delay, n_test, n_epochs):
    """
    Saves automatically each new model execution. Make sure to change:
        - model_name: A new name for your model
        # random_number: Generates a random number to make sure not to overwrite any existing model
    """
    random_number = str(random.randint(1,99999999999))
    model_stats = '_lookBack_'+str(lookBack)+'_delay_'+str(delay)+'_test_'+str(n_test)+'_epochs_'+str(n_epochs)
    model_name = '_modelName_'+'changeName'
    model_random_number = '_'+ random_number
    model.save(save_model_directory+model_stats+model_name+model_random_number)
    print("\n model saved as:", save_model_directory+model_stats+model_name+model_random_number)

def execute_new_model():
    """
    Executes a new model. Make sure to:
        1. Change load_dataset and save_model_directory
        2. Have a valid dataset (Two columns: Year_Month, Sales)
    """
    load_dataset = 'C:\\Users\\ChoudhuryMB\\Desktop\\lstm-test\\alvin-dataset-new.csv'
    save_model_directory = 'C:\\Users\\ChoudhuryMB\\Desktop\\lstm-test\\'
    
    series = read_csv(load_dataset,  header=0, parse_dates = ['Year_Date'], index_col=0, squeeze=True, date_parser=parser)
    
    neurons = 10
    lookBack = 12
    delay = 2
    test_samples = 12
    n_epochs = 10
    n_batch = 1
    learning_rate = 0.01

    # prepare data
    scaler, train, test = prepare_data(series, test_samples, lookBack, delay)
    # fit model
    model = fit_lstm(neurons, train, lookBack, delay, n_batch, n_epochs, learning_rate)
    # make forecasts
    forecasts = make_forecasts(model, n_batch, train, test, lookBack, delay)
    # inverse transform forecasts and test
    forecasts = inverse_transform(series, forecasts, scaler, test_samples+2)
    actual = [row[lookBack:] for row in test]
    actual = inverse_transform(series, actual, scaler, test_samples+2)
    
    # print the forecast value and the actual value
    print_actual_and_forecasts(test_samples, delay, forecasts, actual)
           
    # evaluate forecasts
    evaluate_forecasts(actual, forecasts, lookBack, delay)
    
    #save_model
    model_save(save_model_directory, model, lookBack, delay, test_samples, n_epochs)


def change_date_to_one_column():
    """
    In order for the model to work you must first seperate the dates to:
    Year and month in different columns such as:
        | Year | Month |
        | 2011 |  01   |
        | 2011 |  02   |
    Then execute change_date_to_one_column() which will transform to:
        | Year_Month |
        | 01-01-2011 |
        | 01-02-2011 |
    Change is needed. This is due to incompatibility issues between 
    excel and the program
    """
    def parse(x):
        return datetime.strptime(x, '%Y %m')
    
    # Old dataset to change
    dataset_directory = 'C:\\Users\\ChoudhuryMB\\Desktop\\lstm-test\\alvin-dataset-nojun.csv'
    
    # New dataset information
    new_dataset_directory = 'C:\\Users\\ChoudhuryMB\\Desktop\\lstm-test\\'
    new_dataset_name = 'alvin-dataset-nojun-new-2'
    new_dataset_extension = '.csv'
    
    dataset = read_csv(dataset_directory,  parse_dates = [['Year', 'Date']], index_col=0, date_parser=parse)
    
    dataset.to_csv(new_dataset_directory+new_dataset_name+new_dataset_extension)
    print("Successfully created a reformated dataset")

def main():
    """
    OPTIONS MENU
    Allows to select differents tasks:
        1. Execute model will allow you to train a new model
        2. Change the dataset will allow you to reformat the dataset so it can run using models
    """
    options = 'What would you like to do?'
    options += '\n Enter 1 to execute a new model'
    options += '\n Enter 2 to format the dataset'
    options += '\n Your option: '
    choice = eval(input(options))
    
    if choice == 1:
        print("You selected to execute a new model")
        execute_new_model()
    elif choice == 2:
        print("You decided to reformat the dataset")
        change_date_to_one_column()
    else:
        print('You entered a wrong number')




"""
Execute main function
"""       
if __name__ == '__main__':
    main()