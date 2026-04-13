import os
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
def main():
    df = pd.read_csv("./data/housing.csv")
    print(df.head())
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

    # Train
    model = LinearRegression()
    model.fit(X_train,y_train)

    # Save the model
    os.mkdir("artifacts")
    model_path = os.path.join("artifacts", "model.pkl")
    joblib.dump(model, model_path)



if __name__ == '__main__':
    main()