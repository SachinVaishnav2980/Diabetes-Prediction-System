# app.py

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# import pickle
# model = pickle.load(open('model.pkl', 'rb'))
# print(model.predict([[1, 85, 66, 29, 0, 26.6, 0.351, 31]]))
# Load the trained model

model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('Diabtes-Prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features=[
            float(request.form['pregnancies']),
            float(request.form['glucose']),
            float(request.form['bloodpressure']),
            float(request.form['skinthickness']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['diabetespedigreefunction']),
            float(request.form['age'])
        ]
        prediction = model.predict(np.array(features).reshape(1, -1))[0]
        print("Prediction from model:", prediction)
        output = 'person is diabetic' if prediction == 1 else 'person is not diabetic'
        return render_template('Diabtes-Prediction.html', prediction_text='Prediction: {}'.format(output))
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)

#0 137	40	35	168	43.1	2.288	33