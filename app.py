import uvicorn
from fastapi import FastAPI
import  joblib
from pydantic import BaseModel
from typing import List
import pandas as pd
class PredictionInput(BaseModel):
    features : List[float]

app = FastAPI()

model = joblib.load("./artifacts/model.pkl")

@app.get("/")
def test():
    return {"message":"ML API Is running"}

'''
{
"features" :[5.1,3.5,1.4,0.2]
}
'''
@app.post("/predict")
def predict(data:PredictionInput):
    df = pd.DataFrame([data.features])
    prediction = model.predict(df)
    return {"prediction":int(prediction)}



if __name__ == '__main__':
    uvicorn.run("app:app", host="127.0.0.1", port=8001, reload=True)