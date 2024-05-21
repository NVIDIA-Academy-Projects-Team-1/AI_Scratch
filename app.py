# app.py for attaching frontend


## MODULE IMPORTS ##
import tensorflow as tf
import numpy as np
import pandas as pd
import re
import cv2 as cv
import os
import ollama
import speech_recognition as sr
import matplotlib
import matplotlib.pyplot as plt
import mpld3

from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from flask import Flask, render_template, request, jsonify, Response, redirect, url_for, stream_with_context, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array, load_img


## GLOBAL FIELD ##
app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 10
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
app.config['UPLOAD_FOLDER'] = 'uploads'

data = None
model = None
label = []
type = None
encoded_cols = None

vgg16_model = VGG16(weights = 'imagenet', include_top = True, classes = 1000)

losses = []

matplotlib.use("macOSX")
matplotlib.pyplot.switch_backend('Agg') 

# matplotlib.use('agg')

## FLASK APP ROUTES ##
@app.route("/", methods = ['POST','GET'])
def init():
    return render_template('index.html')


@app.route("/data", methods = ["POST"])
def fetch_data():
    global data
    global upload_img_path, upload_csv_path, image, filename
    data = request.form
    print('==========  fetch_data called  ==========')
    print(data)
    print('==========     end of data     ==========')
    
    if data['type'] == 'number' or data['type'] == 'number-file':
        if data['type'] == 'number-file':
            regexp = r'^[0-9,-.]+$'
            parsed_x = request.files['x_file'].read().decode('utf-8').replace('\n', ',')
            parsed_y = request.files['y_file'].read().decode('utf-8').replace('\n', ',')
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

    elif data['type'] == 'csv-file':
        csv_file = request.files['csv_file']
        filename = secure_filename(csv_file.filename)
        upload_csv_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        csv_file.save(upload_csv_path)
        return process_csv()
        
    
    elif data['type'] == 'image':
        image = request.files['img']
        filename = secure_filename(image.filename)
        upload_img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
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

    print("fetchted audio: ", len(audio_file))

    with open('audio.opus', 'wb') as f:
        f.write(audio_file)

    os.system(f'ffmpeg -y -i "audio.opus" -vn "audio.wav"')

    with sr.AudioFile('audio.wav') as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio, language='ko-KR')
        print("Recognised text: ", text)
        return jsonify({'text': text})
    except sr.UnknownValueError:
        print("Recognition error")
        return jsonify({'error': '인식할 수 없습니다'}), 400


@app.route("/testdata", methods = ["POST"])
def fetch_test_data():
    global label, type, model, encoded_cols
    test_data = request.form

    if test_data['type'] == 'plain':
        x_data = np.array([[float(i)] for i in test_data['x'].split(',')])
        x_data = tf.data.Dataset.from_tensor_slices(x_data)

        def pred():
            for x in x_data:
                pred = model.predict(x, verbose = 0)
                print(pred)
                yield f"{x[0]}에 대한 예측값은 {round(float(pred[0][0]))}입니다.\n"

    elif test_data['type'] == 'csv':
        x = pd.DataFrame(test_data['x'].split(',')).T
        print(x, x.shape)
        
        if type == 'Linear':
            for col in x.columns:
                x[col] = pd.to_numeric(x[col], errors = 'ignore')
            print(x.info())
            objectCol = [x.columns.get_loc(col) for col in x.dtypes[x.dtypes == 'object'].index]
            print("objectcol len: ", len(objectCol))
            if len(objectCol) != 0:
                x = pd.get_dummies(x, prefix = objectCol, columns = objectCol, dtype = int)
                x = x.reindex(columns = encoded_cols)
            @stream_with_context
            def pred():
                try:
                    pred = model.predict(x, verbose = 0)
                    print(pred)
                    yield f"{test_data['x']}에 대한 예측값은 {round(float(pred[0][0]))}입니다.\n"
                except:
                    yield "입력값이 올바르지 않습니다."

        elif type == 'Logistic':
            for col in x.columns:
                x[col] = pd.to_numeric(x[col], errors = 'ignore')
            print(x.info())
            objectCol = [x.columns.get_loc(col) for col in x.dtypes[x.dtypes == 'object'].index]
            print("objectcol len: ", len(objectCol))
            if len(objectCol) != 0:
                x = pd.get_dummies(x, prefix = objectCol, columns = objectCol, dtype = int)
                x = x.reindex(columns = encoded_cols)
            print("x :\n", x)
            @stream_with_context
            def pred():
                try:
                    pred = model.predict(x, verbose = 0)
                    print(pred)
                    print(label)
                    yield f"{test_data['x']}에 대한 예측값은 {label[np.argmax(pred[0])]}입니다.\n"
                except:
                    yield "입력값이 올바르지 않습니다."
    return Response(pred())

@app.route("/losses", methods=['GET'])
def get_losses():
    global losses
    return jsonify(losses)

@app.route("/plot")
def plot_loss():
    return render_template('index.html')

@app.route("/response", methods = ["GET"])
def process_number():
    global model
    global losses
    losses = []
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
            epoch_loss = []
            for x, y in dataset:
                with tf.GradientTape() as tape:
                    logit = model(x, training = True)
                    loss = lossFunc(y, logit)
                
                grad = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grad, model.trainable_weights))
                epoch_loss.append(float(loss))

            avg_loss = np.mean(epoch_loss)
            losses.append(avg_loss)

            print(f"epoch {i + 1} done, loss {float(loss)}")
            yield f"현재 모델은 {i + 1}번째 학습중이며, 목표값과의 차이는 {float(loss):.2f}입니다.\n"
        
        plt.plot(losses)
        plt.savefig('uploads/fig.png')

    return Response(train())



@app.route("/response", methods = ["GET"])
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

@app.route("/response", methods = ["GET"])
def process_csv():
    global label, type, model, encoded_cols, data

    csv_data = pd.read_csv(upload_csv_path, header = None, skiprows = [0])
    csv_data = csv_data.dropna(axis = 0)

    type = 'Linear' if csv_data.iloc[:, -1].dtype == 'float64' else 'Logistic'
    print("Data : ", csv_data.iloc[0], "\nTarget data type : ", csv_data.iloc[:, -1].dtype)

    ## Train Logistic Regression Dataset ##
    if type == 'Logistic':
        x = csv_data.iloc[:, :-1]
        y = csv_data.iloc[:, -1]
        label = csv_data.iloc[:, -1].unique()

        units_1 = int(data['units1']) if data['units1'] != '' else 0
        units_2 = int(data['units2']) if data['units2'] != '' else 0
        units_3 = int(data['units3']) if data['units3'] != '' else 0

        unit_list = [units_1, units_2, units_3]
        activation_list = [data['activation1'], data['activation2'], data['activation3']]
        layer_list = [Dense(units = i, activation = j) for i, j in zip(unit_list, activation_list) if i != 0]

        objectCol = [x.columns.get_loc(col) for col in x.dtypes[x.dtypes == 'object'].index]
        print("object column : ", objectCol)
        
        train_x = pd.get_dummies(x, prefix = objectCol, columns = objectCol, dtype = int)
        encoded_cols = train_x.columns
        train_y = pd.get_dummies(y, dtype = int)

        dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(len(train_x)).batch(32)

        print("data shape: ", train_x.shape, train_y.shape)
        print("x shape : ", train_x.shape[1])

        input = Input(shape = (train_x.shape[1], ), batch_size = 32)
        # x = Dense(32, activation = "relu")(input)
        # x = Dense(64, activation = "relu")(x)
        x = input
        for layer in layer_list:
            x = layer(x)
        output = Dense(train_y.shape[1], activation = "softmax")(x)

        model = Model(inputs = input, outputs = output)
        model.summary()

        optimizer = tf.keras.optimizers.Adam()
        lossFunc = tf.keras.losses.CategoricalCrossentropy()
        acc = tf.keras.metrics.CategoricalAccuracy()

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
                print(f"epoch {i + 1} done, accuracy {float(accuracy) * 100:.4f}%")
                yield f"현재 모델은 {i + 1}번째 학습중이며, 정확도는 {float(accuracy * 100):.2f}% 입니다.\n"
        
        return Response(train())

    ## Train Linear Regression Dataset ##
    elif type == 'Linear':
        x = csv_data.iloc[:, :-1]
        train_y = csv_data.iloc[:, -1]

        units_1 = int(data['units1']) if data['units1'] != '' else 0
        units_2 = int(data['units2']) if data['units2'] != '' else 0
        units_3 = int(data['units3']) if data['units3'] != '' else 0

        unit_list = [units_1, units_2, units_3]
        activation_list = [data['activation1'], data['activation2'], data['activation3']]
        layer_list = [Dense(units = i, activation = j) for i, j in zip(unit_list, activation_list) if i != 0]

        objectCol = [x.columns.get_loc(col) for col in x.dtypes[x.dtypes == 'object'].index]
        print("object column : ", objectCol)
        
        train_x = pd.get_dummies(x, prefix = objectCol, columns = objectCol, dtype = int)

        dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(len(train_x)).batch(32)

        print("data shape: ", train_x.shape, train_y.shape)
        print("x shape : ", train_x.shape[1])

        input = Input(shape = (train_x.shape[1], ), batch_size = 32)
        # x = Dense(32, activation = "relu")(input)
        # x = Dense(64, activation = "relu")(x)
        x = input
        for layer in layer_list:
            x = layer(x)
        output = Dense(1)(x)

        model = Model(inputs = input, outputs = output)
        model.summary()

        optimizer = tf.keras.optimizers.Adam()
        lossFunc = tf.keras.losses.MeanSquaredError()

        def train():
            for i in range(10):
                for x, y in dataset:
                    with tf.GradientTape() as tape:
                        logit = model(x, training = True)
                        loss = lossFunc(y, logit)
                        
                    grad = tape.gradient(loss, model.trainable_weights)
                    optimizer.apply_gradients(zip(grad, model.trainable_weights))
            
                print(f"epoch {i + 1} done, loss {float(loss)}")

                yield f"현재 모델은 {i + 1}번째 학습중이며, 목표값과의 차이는 {float(loss):.2f} 입니다.\n"
        
        return Response(train())

    ## Error classifying ##
    else:
        print("Failed to classify dataset with target type: ", type)
        return jsonify({"alert": "CSV파일 읽기에 실패했습니다."})


@app.route('/uploads/<filename>')
def uploads_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route("/response", methods = ["GET"])
def process_image():
    img = load_img(upload_img_path, target_size = (224, 224))
    img = img_to_array(img)
    img = img.astype('float32')
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = preprocess_input(img)

    pre = vgg16_model.predict(img, verbose = 0)
    result = decode_predictions(pre)[0][0]
    label = result[1].replace('_', ' ').lower()
    acc = result[2]
    print("result : ", result)
    print("original label : ", label)

    response = ollama.chat(model = 'thinkverse/towerinstruct',messages=[
        {
            'role' : 'user',
            'content' : f'Translate following text from English into Korean.\nEnglish: Predicted result of image is "{label}", with accuracy of {float(acc * 100):.2f}%.\nKorean:'
        }
    ])

    print('generated text: ', response['message']['content'], 'acc: %.2f' % float(acc * 100))
    text = f"예측한 사진의 종류는 {response['message']['content'].replace('.', '')}이고, 예측 정확도는 {float(acc * 100):.2f}% 입니다."
    return jsonify({'response': response['message']['content'], 'image_path':upload_img_path})


@app.route("/response", methods=["GET"])
def createResponse():
    content = data['text']

    print("Fetched content: ", content)
    
    response = ollama.chat(model = 'llama3', messages = [
            {
                'role' : 'system',
                'content' : 'You are a helpful AI responding to Korean users. You must create response only in Korean language no matter what.',
            },
            {
                'role' : 'system',
                'content' : 'You must not response to question related to drugs or other sensitive subjetcs. When you receive questions as mentioned before, answer 죄송합니다. 해당 질문에 대한 답변은 해 드릴 수 없습니다.'
            },
            {
                'role' : 'user',
                'content' : f'{content}',
            }
        ],
    )

    if 'I cannot provide' in response['message']['content']:
        original_answer = "죄송합니다. 해당 질문에 대한 답변은 해 드릴 수 없습니다."
    else:
        original_answer = response['message']['content']

    print("Original Answer : ", original_answer)

    
    return jsonify({'response': original_answer})      


## RUN FLASK APP ##
if __name__ == "__main__":
    app.run(debug = True)
    # app.run(host = '192.168.1.122', port = 5500, debug = True)
    