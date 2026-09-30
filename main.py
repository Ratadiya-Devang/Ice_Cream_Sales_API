import pandas as pd
import joblib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = joblib.load("model.pkl")


class PredictionRequest(BaseModel):
    temperature: float = Field(
        ...,
        description="Temperature in Celsius"
    )


@app.get("/")
def home():
    return {
        "msg": "Temperature prediction API is running"
    }


@app.post("/predict")
def predict(data: PredictionRequest):

    input_data = pd.DataFrame({
        "Temperature_C": [data.temperature]
    })

    prediction = model.predict(input_data)

    sales = int(prediction[0])

    return {
        "sales": sales
    }