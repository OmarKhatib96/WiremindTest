

from const import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


def model(input):
    model = Sequential()
    model.add(Dense(units=64, input_shape=(input.shape[1],)))
    model.add(Dropout(DO))

    model.add(Dense(units=18,activation='relu'))
    # Add an output layer
    model.add(Dense(units=1))

    # Compile the model
    opt = keras.optimizers.RMSprop(lr=learning_rate)
    model.compile(optimizer = opt, loss = 'mse')
    return model

