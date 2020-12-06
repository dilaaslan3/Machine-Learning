import numpy as np
from sklearn.model_selection import train_test_split
from knn import KNN
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import argparse


def main():
    df = pd.read_csv(f".\Data\{args.dataset}")

    X = np.array(df.iloc[:, :-1])
    y = np.array(df.iloc[:, -1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn = KNN(X_train, y_train, k=args.k)

    if args.type == "clf":
        y_pred = knn.predict(X_test, knn_type="clf")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

    elif args.type == "reg":
        y_pred = knn.predict(X_test, knn_type="reg")
        mse = np.mean((y_test-y_pred)**2)
        print(mse)

    else:
        return print("Undefined knn type")

    accuracy = np.mean(y_pred == y_test)
    print(accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", type=str, required=True, help="dataset")
    parser.add_argument('-k', type=int, required=False, default=3, help="number of nearest neighbors")
    parser.add_argument("-type", type=str, required=True, help="identify the problem type, for classification:clf, for regression:reg ")
    args = parser.parse_args()
    main()
