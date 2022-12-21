#https://www.tensorflow.org/guide/core/mlp_core
#https://github.com/tensorflow/docs/blob/master/site/en/guide/core/mlp_core.ipynb
# Use seaborn for countplot
#!pip install -q seaborn

import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import tempfile
import os
# Preset Matplotlib figure sizes.
matplotlib.rcParams['figure.figsize'] = [9, 6]

from keras import datasets, layers, models

import tensorflow as tf
import tensorflow_datasets as tfds
print(tf.__version__)
# Set random seed for reproducible results
tf.random.set_seed(22)

train_data, val_data, test_data = tfds.load("mnist",
                                            split=['train[10000:]', 'train[0:10000]', 'test'],
                                            batch_size=128, as_supervised=True)
