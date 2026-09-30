import pandas as pd 
import joblib
from fastapi import FastAPI 
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel,Field

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins = ["http://localhost:5173"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

model = joblib.load("model.pkl")

class TempratureInput(BaseModel):
    temprature:float


@app.get("/")
def home():
    return {"msg":"temprature prediction api is running"}

@app.post("/predict")
def predict(data:TempratureInput):
    temprature_value = data.temprature


    input_data = pd.DataFrame({
        "Temperature_C":[temprature_value]
    }
    )

    prediction = model.predict(input_data)

    return {"temprature":temprature_value,
            "predict":float(prediction[0])}