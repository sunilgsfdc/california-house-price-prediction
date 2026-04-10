import pickle

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os
import joblib
from pathlib import Path
import json



def main():
    iris = load_iris()
    x,y = iris.data, iris.target
    print(x)
    print(y)
    print(iris.target_names)

    # model is a mathematical equation
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

    model = LogisticRegression(max_iter=200) # algorithm
    model.fit(X_train, y_train) # training the model

    if not Path('artifacts').exists():
        os.mkdir('artifacts')

    model_path = os.path.join('artifacts', 'model.pkl')
    joblib.dump(model, model_path)

    acc = model.score(X_test, y_test) # checking for accuracy of the data
    metrics = {'accuracy': float(acc)}

    with open(os.path.join('artifacts', 'metrics.json'), 'w') as f:
        json.dump(metrics, f)

    print(acc)

if __name__ == '__main__':
    main()
