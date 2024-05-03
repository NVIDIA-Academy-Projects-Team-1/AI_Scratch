# app.py for attaching frontend

import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Conv2D
from flask import Flask, render_template, request, jsonify, Response, redirect, url_for

import time

app = Flask(__name__)
data = None
model = None


@app.route("/", methods = ['POST','GET'])
def init():
    return render_template('index.html')


@app.route("/data", methods = ["POST"])
def fetch_data():
    global data
    data = request.form
    
    if data['type'] == 'number':
        return redirect(url_for('process_number'))
    elif data['type'] == 'image':
        process_image()
    elif data['type'] == 'text':
        process_text()
    else:
        return jsonify({'msg':"invalid input type"})
    
    return jsonify({'msg':"server routing error occurred"})



@app.route("/response", methods = ["GET"])
def process_number():
    print('==========process_number called==========')
    x_data = np.array([[float(i)] for i in data['x'].split(',')])
    print(x_data.shape)
    y_data = np.array([[float(i)] for i in data['y'].split(',')])
    units = int(data['units'])
    activation = data['activation']

    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))

    input = Input(shape = (1, ))
    x = Dense(units = units, activation = activation)(input)
    output = Dense(1)(x)

    model = Model(inputs = input, outputs = output)
    optimizer = keras.optimizers.SGD(learning_rate = 1e-3)
    lossFunc = keras.losses.MeanSquaredError()
    
    def train():
        for i in range(10):
            for x, y in dataset:
                #print("x : ", x)
                with tf.GradientTape() as tape:
                    logit = model(x, training = True)
                    loss = lossFunc(y, logit)
                
                grad = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grad, model.trainable_weights))
            print(f"epoch {i + 1} done")
            yield f"epoch: {i + 1}, loss: {float(loss)}\n"
            
    return Response(train())



def process_image(data):
    pass

def process_text(data):
    pass





if __name__ == "__main__":
    app.run(host = '127.0.0.1', debug = True)


