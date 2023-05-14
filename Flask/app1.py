from flask import Flask, request, jsonify
import pickle
from flask_ngrok import run_with_ngrok
import numpy as np
import pandas as pd

diabetes = pickle.load(open("Flask\diabetes_model.pkl",'rb'))

app = Flask(__name__)
run_with_ngrok(app)

@app.route('/mrs' , methods=['POST'])
def diabetes_prediction():
    # Get the input data from the request
    input_data = request.get_json()
    
    # Extract the input values from the data
    pregnancies = int(input_data['pregnancies'])
    glucose = int(input_data['glucose'])
    blood_pressure = int(input_data['blood_pressure'])
    skin_thickness = int(input_data['skin_thickness'])
    insulin = int(input_data['insulin'])
    bmi = float(input_data['bmi'])
    diabetes_pedigree_function = float(input_data['diabetes_pedigree_function'])
    age = int(input_data['age'])

    # Create a numpy array with the input values
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])

    # Reshape the input data to match the expected shape of the model
    input_data_reshaped = input_data.reshape(1,-1)

    # Make the prediction using the loaded model
    prediction = diabetes.predict(input_data_reshaped)

    # Return the predicted outcome as a JSON response
    if (prediction[0] == 0):
        return jsonify({'prediction': 'Diabetes Negative'})
    else:
        return jsonify({'prediction': 'Diabetes Positive'})
    
if __name__ == '__main__':
    app.run()