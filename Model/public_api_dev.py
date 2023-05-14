from fastapi import FastAPI
from pydantic import BaseModel
import pickle 
import json
import uvicorn
from pyngrok import ngrok
from fastapi.middleware.cors import CORSMiddleware
import nest_asyncio

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class model_input(BaseModel):
    Pregnancies : int
    Glucose : int
    BloodPressure : int
    SkinThickness : int
    Insulin : int
    BMI : float
    DiabetesPedigreeFunction : float
    Age :int

model = pickle.load(open('diabetes_model.sav','rb'))

@app.post('/diabetes_prediction')
def diab_predd(input_parameters : model_input):
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)

    preg = input_dictionary['Pregnancies']
    glu = input_dictionary['Glucose']
    bp = input_dictionary['BloodPressure']
    skin = input_dictionary['SkinThickness']
    insulin = input_dictionary['Insulin']
    bmi = input_dictionary['BMI']
    dpf = input_dictionary['DiabetesPedigreeFunction']
    age = input_dictionary['Age']

    input_list = [preg,glu,bp,skin,insulin,bmi,dpf,age]

    prediction = model.predict([input_list])

    if (prediction[0]==0):
        return'The person is not diabetic'
    else:
        return 'The person is diabetic'
    
ngrok_tunnel = ngrok.connect(8000)
print("Public URL:",ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app,port=8000) 
public_url__ = ngrok_tunnel.public_url 
