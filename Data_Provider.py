import pandas as pd
import numpy as np
import os


class DataProvider:

    def __init__(self, filename, out_folder):
        self.filename = filename
        self.out_folder = out_folder

    def read(self):
        df = pd.read_csv(self.filename)

        # null checkpoint
        print(df.isnull().sum())

        print(df.info())
        print("Categorical features distribution: \n", df.describe(include=[object]))
        print("\nNumeric features distribution: \n", df.describe())

        return df

    def one_hot_encoding(self, df):
        categorical_features = list((df.select_dtypes(include=['object'])).columns)
        X = pd.get_dummies(df, columns=categorical_features, drop_first=True)
        return X

    def split(self, df, train_ratio, target):

        X_train, X_validation, X_test = np.split(df.sample(frac=1), [int(train_ratio*len(df)), int((((1-train_ratio)/2)+train_ratio)*len(df))])

        y_train, y_validation, y_test = X_train[target].to_frame(), X_validation[target].to_frame(), X_test[target].to_frame()

        X_train = X_train.drop([target], axis=1)
        X_validation = X_validation.drop([target], axis=1)
        X_test = X_test.drop([target], axis=1)

        return X_train, X_validation, X_test, y_train, y_validation, y_test

    def normalize(self, train, validation, test, x_or_y):

        # Z-score = (value - mean) / std
        train_df_mean = train.mean()
        train_df_std = train.std()

        if x_or_y is True:
            self.save(train_df_mean, train_df_std, True)
        else:
            self.save(train_df_mean, train_df_std, False)

        train_df_norm = (train - train_df_mean) / train_df_std
        val_df_norm = (validation - train_df_mean) / train_df_std
        test_df_norm = (test - train_df_mean) / train_df_std
        return train_df_norm, val_df_norm, test_df_norm

    def save(self, train_mean, train_std, x_or_y):
        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)

        if x_or_y is True:
            train_mean.to_csv(self.out_folder + os.sep + "features_train_mean.csv")
            train_std.to_csv(self.out_folder + os.sep + "features_train_std.csv")
        else:
            train_mean.to_csv(self.out_folder + os.sep + "target_train_mean.csv")
            train_std.to_csv(self.out_folder + os.sep + "target_train_std.csv")
