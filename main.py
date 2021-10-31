from flask import Flask,request,jsonify
import pickle
import numpy as np
import sklearn


model = pickle.load(open('model1.pkl','rb'))


main = Flask(__name__)
@main.route('/')

def home():
    return "Hello World"

@main.route('/predict',methods = ['POST'])
def predict():
    cgpa = request.form.get('cgpa')
    iq = request.form.get('iq')
    p_s = request.form.get('p_s')

    input_query = np.array([[cgpa,iq,p_s]])

    result = model.predict(input_query)[0]

    return jsonify({'placement':str(result)})


if __name__ == '__main__':
    main.run(debug=True)



