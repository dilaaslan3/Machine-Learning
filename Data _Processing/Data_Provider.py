import pandas as pd
import numpy as np


class DataProvider:

    def read(self, path):
        df = pd.read_csv(path)
        df = df.drop(['car_ID', 'CarName'], axis=1)

        # null checkpoint
        # print(df.isnull().sum())

        # print(df.info())
        # print("Categorical features distribution: \n", df.describe(include=[object]))
        # print("\nNumeric features distribution: \n", df.describe())

        return df

    def one_hot_encoding(self, df):
        categorical_features = list((df.select_dtypes(include=['object'])).columns)
        X = pd.get_dummies(df, columns=categorical_features, drop_first=True)
        return X

    def split(self, df, train_ratio):

        X_train, X_validation, X_test = np.split(df.sample(frac=1), [int(train_ratio*len(df)), int((((1-train_ratio)/2)+train_ratio)*len(df))])

        y_train, y_validation, y_test = X_train["price"], X_validation["price"], X_test["price"]

        X_train = X_train.drop(["price"], axis=1)
        X_validation = X_validation.drop(["price"], axis=1)
        X_test = X_test.drop(["price"], axis=1)

        return X_train, X_validation, X_test, y_train, y_validation, y_test

    def normalize(self, train, validation, test):

        # Z-score = (value - mean) / std

        train_df_mean = train.mean()
        train_df_std = train.std()

        train_df_norm = (train - train_df_mean) / train_df_std
        val_df_norm = (validation - train_df_mean) / train_df_std
        test_df_norm = (test - train_df_mean) / train_df_std

        return train_df_mean, train_df_std, train_df_norm, val_df_norm, test_df_norm

    def save(self, train_mean, train_std):
        train_mean.to_csv("train_mean.csv")
        train_std.to_csv("train_std.csv")
