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
    # Extract the input values from the data
    pregnancies = int(request.args.get('pregnancies'))
    glucose = int(request.args.get['glucose'])
    blood_pressure = int(request.args.get['blood_pressure'])
    skin_thickness = int(request.args.get['skin_thickness'])
    insulin = int(request.args.get['insulin'])
    bmi = float(request.args.get['bmi'])
    diabetes_pedigree_function = float(request.args.get['diabetes_pedigree_function'])
    age = int(request.args.get['age'])

    # Create a numpy array with the input values
    input_data = np.array([[pregnancies,glucose,blood_pressure,skin_thickness,insulin,bmi,diabetes_pedigree_function,age]])

    # Reshape the input data to match the expected shape of the model
    input_data_reshaped = input_data.reshape(1,-1)
    # Make the prediction using the loaded model
    prediction = diabetes.predict(input_data_reshaped)

    # Return the predicted outcome as a JSON response
    return jsonify(prediction)
    
if __name__ == '__main__':
    app.run()