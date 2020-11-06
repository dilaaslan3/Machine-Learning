import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import argparse
from linear_reg import LinearRegression


def main():

    dataset = pd.read_csv(f'././{args.file_name}')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    regressor = LinearRegression(float(args.lr), int(args.n_iter))
    regressor.fit(X_train, y_train)
    predicted = regressor.predict(X_test)
    mse_value, mae_value, msqrte_value, rsqrte_value = regressor.score(
        y_test, predicted)
    print("Mean Squared Error score is: ", mse_value)
    print("Mean Absolute Error score is: ", mae_value)
    print("Root Mean Absolute Error score is: ", msqrte_value)
    print("R-Squared Error score is: ", rsqrte_value)

    y_pred_line = regressor.predict(X)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train, y_train, color='b', s=10)
    plt.scatter(X_test, y_test, color='r', s=10)
    plt.plot(X, y_pred_line, color='black', linewidth=2, label="Prediction")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-file_name", help="file_name")
    parser.add_argument('-lr', help="learning rate, float")
    parser.add_argument("-n_iter", help="number of iterations, int")
    args = parser.parse_args()
    main()
