"""
CLASIFICATION MODEL
OUTCOME MUST BE 0 or 1
"""

from time import time

import numpy
import pandas

import keras
from keras.models import Sequential
from keras import layers, optimizers

# Tell keras to use tensorflow as back-end
os.environ['KERAS_BACKEND'] = 'tensorflow'

import pandas as pd
import seaborn as sns

from keras.callbacks import TensorBoard

df = pd.read_csv('C:\\Users\\choudhuryMB\\Desktop\\WTW\\PMI_dataset_small.csv',
                 header=None,
                 sep=',')

df.columns = ['year', 'month', 'existing_book', 'new_business', 'lost_business', 'pred']

print("Data Description")
df.describe()

def preprocess_features(df):
    
    selected_features = df[['year', 'existing_book']]
    return selected_features

def preprocess_targets(df):
    
    output_target = df['pred']
    return output_target

training_examples = preprocess_features(df.head(190))
training_targets = preprocess_targets(df.head(190))

test_examples = preprocess_features(df.tail(30))
test_targets = preprocess_targets(df.tail(30))

print("Training examples summary:")
print(training_examples.describe())
print("Validation examples summary:")
print(test_examples.describe())
print("Training targets summary:")
print(training_targets.describe())
print("Validation targets summary:")
print(test_targets.describe())


"""
#ANOTHER WAY OF SPLITTING THE DATA
train_index = df.sample(frac=0.8, random_state=21).index
X_train, y_train = df.iloc[train_index][['year', 'existing_book']], df.iloc[train_index]['pred']
X_test, y_test = df.drop(index=train_index)[['year', 'existing_book']], df.drop(index=train_index)['pred']
"""
sns.lmplot(data=df, x='year', y='existing_book', hue='pred', fit_reg=False);


def make_model():
    model = Sequential(name='Revenue predicition based on years')
    
    model.add(layers.Dense(name='1', units=3, input_dim=2))
    model.add(layers.Activation(name='relu1', activation='relu'))
    
    model.add(layers.Dense(name='2', units=2))
    model.add(layers.Activation(name='relu2', activation='relu'))
    
    model.add(layers.Dense(name='3', units=1))
    model.add(layers.Activation(name='sigmoid', activation='sigmoid'))
    
    model.summary()
    
    return model

model = make_model()

model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Check TensorBoard for Training Loss and Validation loss")
tensorboard = TensorBoard(log_dir='C:\\Users\\choudhuryMB\\Desktop\\WTW\\{}'.format(time()))

print("Starting to train the model...")
summary = model.fit(
        training_examples, training_targets,
        batch_size=1,
        epochs=50,
        validation_split=0.1,
        verbose=0,
        callbacks=[tensorboard])

score = model.evaluate(test_examples, test_targets, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
