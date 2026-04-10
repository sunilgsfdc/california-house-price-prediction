import uvicorn
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("./artifacts/model.pkl")

@app.get("/")
def test_api():
    return {"message":"Test API"}

@app.post("/predict")
def predict(data:dict):
    df = pd.DataFrame[data["features"]]
    prediction = model.predict(df)
    return {"prediction":float(prediction[0])}

if __name__ = '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=8000)