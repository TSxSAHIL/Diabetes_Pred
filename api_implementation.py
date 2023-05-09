import json
import requests

url = 'public_url__/diabetes_prediction'

input_data_model = {

    'Pregnancies' : 0,
    'Glucose': 148,
    'BloodPressure': 72,
    'SkinThickness' : 35,
    'Insulin' : 0,
    'BMI' : 33.6,
    'DiabetesPedigreeFunction' : 0.627,
    'Age' : 60

}

input_json = json.dumps(input_data_model)

response = requests.post(url, data=input_json)

print(response.text)