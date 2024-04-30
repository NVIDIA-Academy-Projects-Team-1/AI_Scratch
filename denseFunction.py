import sys
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

def firstdenseFunction(units,input_shape,activation):
    model=Sequential[
        Dense(units=units,input_shape=input_shape,activation=activation)
    ]


if __name__ == '__main__':
    if sys.argv[2] == []:
        firstdenseFunction(sys.argv[1],sys.argv[3])
    else:
        firstdenseFunction(sys.argv[1],sys.argv[2],sys.argv[3])
    