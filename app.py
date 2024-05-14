# app.py for attaching frontend


## MODULE IMPORTS ##
import tensorflow as tf
import numpy as np
import re
import json
import torch
import math
import cv2 as cv
import time
import os
import ollama
import speech_recognition as sr
import pyaudio
import soundfile
import io


from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout, MaxPooling2D, Flatten
from flask import Flask, render_template, request, jsonify, Response, redirect, url_for, stream_with_context, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image


## GLOBAL FIELD ##
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 10
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
app.config['UPLOAD_FOLDER'] = 'uploads'

data = None
model = None

vgg16_model = VGG16(weights = 'imagenet', include_top = True, classes = 1000)


## FLASK APP ROUTES ##
@app.route("/", methods = ['POST','GET'])
def init():
    return render_template('index.html')


@app.route("/data", methods = ["POST"])
def fetch_data():
    global data
    global upload_img_path, image, filename
    data = request.form
    print('==========  fetch_data called  ==========')
    print(data)
    print('==========     end of data     ==========')
    
    if data['type'] == 'number' or data['type'] == 'number-file':
        if data['type'] == 'number-file':
            regexp = r'^[0-9,]+$'
            parsed_x = request.files['x_file'].read().decode('utf-8')
            parsed_y = request.files['y_file'].read().decode('utf-8')
            data = request.form.to_dict()

            if re.match(regexp, parsed_x) is None or re.match(regexp, parsed_y) is None:
                return jsonify({"alert" : "파일에는 콤마(,)로 구분된 숫자만 들어있어야 합니다."})
            data['x'] = parsed_x
            data['y'] = parsed_y
            return process_number()
        if data['reg_type'] == 'linear':
            return process_number()
        else:
            return process_number_logistic()
    
    elif data['type'] == 'image':
        image = request.files['img']
        # print('-------------test',image)
        filename = secure_filename(image.filename)
        upload_img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # print(f'---------------------{upload_img_path}-------{image}')
        image.save(upload_img_path)
        return process_image()
    
    elif data['type'] == 'text':
        return createResponse()
    
    else:
        return jsonify({'msg':"invalid input type"})


@app.route("/audio_to_text", methods=["POST"])
def audio_to_text():
    recognizer = sr.Recognizer()
    audio_file = request.files['audio'].read()

    # with open('audio.wav', 'wb') as f:
    #     f.write(audio_file)
    
    # data, samplerate = soundfile.read('/Users/jaeh/Documents/NVIDIA_AI_Academy/AI_Scratch/audio.wav')
    # soundfile.write(audio_file, data, samplerate, subtype = 'PCM_16')

    with io.BytesIO(audio_file) as f:
        data, samplerate = soundfile.read(f)
        soundfile.write('audio.wav', data, samplerate)
    
    with sr.AudioFile('audio.wav') as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio, language='ko-KR')
        print('인식된 음성은 : {}'.format(text))
        return jsonify({'text': text})
    except sr.UnknownValueError:
        print("Recognition error")
        return jsonify({'error': '인식할 수 없습니다'}), 400


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

    if x_data.size != y_data.size:
        return jsonify({"alert" : "파일의 목표값과 시작값의 개수가 다릅니다."})

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

    optimizer = keras.optimizers.Adam()
    lossFunc = keras.losses.MeanSquaredError()
            
    def train():
        for i in range(10):
            for x, y in dataset:
                with tf.GradientTape() as tape:
                    logit = model(x, training = True)
                    loss = lossFunc(y, logit)
                
                grad = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grad, model.trainable_weights))
            print(f"epoch {i + 1} done, loss {float(loss)}")
            yield f"현재 모델은 {i + 1}번째 학습중이며, 목표값과의 차이는 {float(loss):.2f}입니다.\n"
            
    return Response(train())


@app.route("/responser", methods = ["GET"])
def process_number_logistic():
    global model
    x_data = np.array([[float(i)] for i in data['x_log'].split(',')])
    y_data = np.array([[float(i)] for i in data['y_log'].split(',')])
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
    output = Dense(units = data['class'], activation = 'sigmoid')(x)
    model = Model(inputs = input, outputs = output)
    model.summary()

    optimizer = keras.optimizers.Adam()
    lossFunc = keras.losses.SparseCategoricalCrossentropy(from_logits = True)
    acc = keras.metrics.SparseCategoricalAccuracy()
    
    def train():
        for i in range(10):
            for x, y in dataset:
                with tf.GradientTape() as tape:
                    logit = model(x, training = True)
                    loss = lossFunc(y, logit)
                
                grad = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grad, model.trainable_weights))
                acc.update_state(y, logit)
            accuracy = acc.result()
            print(f"epoch {i + 1} done, accuracy {float(accuracy)}")
            yield f"현재 모델은 {i + 1}번째 학습중이며, 정확도는 {float(accuracy * 100):.2f}% 입니다.\n"
            
    return Response(train())


@app.route('/uploads/<filename>')
def uploads_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)


@app.route("/response", methods = ["GET"])
def process_image():
    img=load_img(upload_img_path,target_size=(224,224))
    img=img_to_array(img)
    img=img.astype('float32')
    img=img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
    img=preprocess_input(img)

    pre=vgg16_model.predict(img, verbose = 0)
    label=decode_predictions(pre)
    label=label[0][0]
    result=(f' 예측한 사진의 종류는 {label[1]} 이고 예측 정확도는 {float(label[2]*100):.2f} 입니다.')

    content = result

    response = ollama.chat(model = 'llama3',messages=[
        {
            'role' : 'system',
            'content' : 'You are a helpful AI responding to Korean users. You must create response in Korean language no matter what.',
        },
        {
            'role' : 'system',
            'content' : "Translate English word, which is from imagenet dataset label, to Korean in given content. You must reply as '예측한 사진의 종류는 <your_translation>이고 예측 정확도는 <accuracy_in_given_content>% 입니다."
        },
        {
            'role' : 'user',
            'content' : f'{content}',
        }
    ])
    print('generated text: ', response['message']['content'], 'acc: %.2f' % float(label[2]*100))

    return jsonify({'response': response['message']['content'], 'image_path':upload_img_path})
    # return jsonify({'response':result, 'image_path':upload_img_path})


@app.route("/response", methods=["GET"])
def createResponse():
    content = data['text']
    
    response = ollama.chat(model = 'llama3', messages = [
            {
                'role' : 'system',
                'content' : 'You are a helpful AI responding to Korean users. You must create response in Korean language no matter what.',
            },
            {
                'role' : 'system',
                'content' : 'You must not response to question related to drugs or other sensitive subjetcs. When you receive questions as mentioned before, answer "죄송합니다. 해당 질문에 대한 답변은 해 드릴 수 없습니다.".'
            },
            {
                'role' : 'user',
                'content' : f'{content}',
            }
        ],
    )

    if response['message']['content'].startswith('I cannot provide'):
        original_answer = "죄송합니다. 해당 질문에 대한 답변은 해 드릴 수 없습니다."
    else:
        original_answer = response['message']['content']

    print("Original Answer : ", original_answer)
    return jsonify({'response': original_answer})


## RUN FLASK APP ##
if __name__ == "__main__":
    app.run(host = '192.168.0.3', port = 5500, debug = True)
    