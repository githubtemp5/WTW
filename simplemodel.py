# import the library will use to manipulate the data

from __future__ import print_function

import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset


# stablish the access to the data and convert to dataframe
df = pd.read_csv('C:\\Users\ChoudhuryMB\\Documents\\ds\\converted-data.csv',
                     encoding = "ISO-8859-1",
                     sep=',',
                     error_bad_lines=False,
                     index_col=False,
                     dtype='unicode')

df = df.reindex(np.random.permutation(df.index))
