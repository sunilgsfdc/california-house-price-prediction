import json
import sys
from pathlib import Path
import joblib
import numpy as np

MODEL_PATH =  Path("artifacts/model.pkl")
if __name__ == '__main__':
    features = [5.1,3.5,1.4,0.2]
    model = joblib.load(MODEL_PATH)
    print(features)
    X = np.array(features).reshape(1,-1)
    print(X)
    pred = model.predict(X)
    print(json.dumps(({"prediction":pred.tolist()})))