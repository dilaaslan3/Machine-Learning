import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
import argparse
from linear_reg import LinearRegression

def main():

    dataset = pd.read_csv(f'././{args.file_name}')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    predicted = regressor.predict(X_test)
    mse_value = regressor.score(y_test, predicted)
    print("score is: ", mse_value)

    y_pred_line = regressor.predict(X)
    plt.figure(figsize=(8,6))
    plt.scatter(X_train, y_train, color='b', s=10)
    plt.scatter(X_test, y_test, color='r', s=10)
    plt.plot(X, y_pred_line, color='black', linewidth=2, label="Prediction")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-file_name", help="file_name")
    args = parser.parse_args()
    main()





