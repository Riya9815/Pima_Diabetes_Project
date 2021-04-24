import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import scipy
app = Flask(__name__)
model = pickle.load(open('diabetes_logistic.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    preg = int(request.form['pregnancies'])
    glucose = int(request.form['glucose'])
    bp = int(request.form['loodpressure'])
    st = int(request.form['skinthickness'])
    dia = int(request.form['DiabetesPedigreeFunction'])


    data = np.array([[preg, glucose, bp, st, dia]])
    output = model.predict(data)
    #print(my_prediction)
    if output > 0.5:
        return render_template('result.html',pred=f'You have chance of having diabetes.\nProbability of having Diabetes is {output}')
    else:
        return render_template('result.html', pred=f'You are safe.\n Probability of having diabetes is {output}')
if __name__ == "__main__":
    app.run(debug=True)