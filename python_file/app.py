from flask import Flask,request,jsonify
from keras.models import Sequential
from keras.layers import Dense


app = Flask(__name__)

@app.reoute("/",methods=['POST','GET'])

def denseFunc():
    units=request.form.get('units_val',type=int)
    activation=request.form.get('act_val',type=str)
    x_val=request.form.get('x_val',type=list)
    y_val=request.form.get('y_val',type=list)

    for i in range(1,len(x_val)):
        x_train=[]
        i+=1
        return x_train.append(x_val[i])

    for j in range(1,len(y_val)):
        y_train=[]
        j+=1
        return y_train.append(y_val[j])

    model=Sequential[
        Dense(units=units,input_dim=x_val.shape[1],activation=activation)
    ]
    
    model.compile(loss='mse',metrics=['accuracy'])
    model.fit(x_train,y_train,epochs=10)

if __name__=='__main__':
    app.run(debug=True)


