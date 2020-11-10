import pandas as pd
import argparse
from sklearn.model_selection import train_test_split


def read(path):

    df = pd.read_csv(path)
    print(df.head())
    print("Shape of the dataset: \n", df.shape)
    print("Information of the dataset: \n", df.info())

    # null checkpoint
    print(df.isnull().sum())
    return df


def one_hot_encoding(df):
    categorical_features = list((df.select_dtypes(include=['object'])).columns)
    X = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    return X


def split(df):
    X = df.drop(["price"], axis=1)
    y = df["price"]

    # TO DO: split data from scratch
    return train_test_split(X, y, test_size=0.2, random_state=42)


def normalize():
    pass


def main():
    df = read(f'./{args.dataset}')
    X = one_hot_encoding(df)
    X_train, X_test, y_train, y_test = split(X)
    normalize()


if __name__ == "__main__":
    data_parser = argparse.ArgumentParser()
    data_parser.add_argument("-dataset", type=str, required=True, help="Enter the dataset file name.")
    args = data_parser.parse_args()
    main()
