import sys
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

def denseFunc(units,input_dim,activation):
    model = Sequential[
        Dense(units=units,input_dim=input_dim,activation=activation)
    ]
    print('test')


if __name__ == '__main__':
    denseFunc(sys.argv[1],sys.argv[2],sys.argv[3])