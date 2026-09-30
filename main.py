import pandas as pd 
import joblib
from fastapi import FastAPI 
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel,Field

app = FastAPI()



model = joblib.load("model.pkl")




@app.get("/")
def home():
    return {"msg":"temprature prediction api is running"}

@app.post("/predict")
def predict(data:float):



    input_data = pd.DataFrame({
        "Temperature_C":[data]
    }
    )

    prediction = model.predict(input_data)
    sales = int(prediction[0])
    return {"sales":sales}