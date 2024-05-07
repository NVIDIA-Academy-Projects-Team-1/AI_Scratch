# app.py for attaching frontend


## MODULE IMPORTS ##
import tensorflow as tf
import numpy as np
import json
import torch
import math
import cv2
import time
import os

from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout, MaxPooling2D, Flatten
from flask import Flask, render_template, request, jsonify, Response, redirect, url_for, stream_with_context
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from tensorflow.keras.preprocessing.image import img_to_array


## GLOBAL FIELD ##
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 10
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
app.config['UPLOAD_FOLDER'] = 'uploads'
data = None
model = None

tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token = '</s>', eos_token = '</s>', unk_token = '<unk>',
            pad_token = '<pad>', mask_token = '<mask>')

text_model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")

vgg16_model = VGG16(weights = 'imagenet', include_top = True, classes=1000)


## FLASK APP ROUTES ##
@app.route("/", methods = ['POST','GET'])
def init():
    return render_template('index.html')


@app.route("/data", methods = ["POST"])
def fetch_data():
    global data
    data = request.form
    
    if data['type'] == 'number':
        return process_number()
    elif data['type'] == 'image':
        image = request.files.get('img')
        filename = secure_filename(image.filename)
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        return process_image()
    elif data['type'] == 'text':
        return generate_response()
    else:
        return jsonify({'msg':"invalid input type"})


@app.route("/testdata", methods = ["POST"])
def fetch_test_data():
    test_data = request.form

    x_data = np.array([[float(i)] for i in test_data['x'].split(',')])
    x_data = tf.data.Dataset.from_tensor_slices(x_data)

    def pred():
        for x in x_data:
            pred = model.predict(x, verbose = 0)
            print(pred)
            yield f"{int(x[0])}에 대한 예측값은 {round(float(pred[0][0]))}입니다.\n"
    return Response(pred())


@app.route("/response", methods = ["GET"])
def process_number():
    global model
    print('==========process_number called==========')
    print(data)
    print('==========     end of data     ==========')
    x_data = np.array([[float(i)] for i in data['x'].split(',')])
    y_data = np.array([[float(i)] for i in data['y'].split(',')])
    units_1 = int(data['units1']) if data['units1'] != '' else 0
    units_2 = int(data['units2']) if data['units2'] != '' else 0
    units_3 = int(data['units3']) if data['units3'] != '' else 0

    unit_list = [units_1, units_2, units_3]
    activation_list = [data['activation1'], data['activation2'], data['activation3']]
    layer_list = [Dense(units = x, activation = y) for x, y in zip(unit_list, activation_list) if x != 0]

    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))

    input = Input(shape = (1, ))
    x = input
    for layer in layer_list:
        x = layer(x)
    output = Dense(1)(x)
    model = Model(inputs = input, outputs = output)
    model.summary()

    optimizer = keras.optimizers.SGD(learning_rate = 1e-3)
    lossFunc = keras.losses.MeanSquaredError()
    
    def train():
        for i in range(10):
            for x, y in dataset:
                #print("x : ", x[0], "y : ", y[0])
                with tf.GradientTape() as tape:
                    logit = model(x, training = True)
                    loss = lossFunc(y, logit)
                
                grad = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grad, model.trainable_weights))
            print(f"epoch {i + 1} done, loss {float(loss)}")
            yield f"현재 모델은 {i + 1}번째 학습중이며, 예측값과의 차이는 {float(loss):.0f}입니다.\n"
            
    return Response(train())


@app.route("/response", methods = ["GET"])
def process_image():
    # image=data['img']
    # image=img_to_array(image)
    # image=image.astype(float)
    # image=image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
    # image=preprocess_input(image)
    # pre=vgg16_model.predict(image)
    # label=decode_predictions(pre)
    # label=label[0][0]
    # return Response(print('%s (%.2f%%)'%(label[1],label[2]*100)))
    pass


@app.route("/response", methods=["GET"])
def generate_response():
    question = "근육을 키우려면?"
    input_ids = tokenizer.encode(question, return_tensors='pt')
    gen_ids = text_model.generate(input_ids,
                             max_length=128,
                             repetition_penalty=2.0,
                             pad_token_id=tokenizer.pad_token_id,
                             eos_token_id=tokenizer.eos_token_id,
                             bos_token_id=tokenizer.bos_token_id,
                             use_cache=True)
    generated = tokenizer.decode(gen_ids[0])
    return f"{generated}"


## RUN FLASK APP ##
if __name__ == "__main__":
    app.run(host = '192.168.0.3', port = 5500, debug = True)
    