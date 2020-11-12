import argparse
from Data_Provider import DataProvider


def main():
    provider = DataProvider()
    df = provider.read(f'./{args.dataset}')
    X = provider.one_hot_encoding(df)
    X_train, X_validation, X_test, y_train, y_validation, y_test = provider.split(X, args.train_ratio)
    X_train_df_mean, X_train_df_std, X_train_norm, X_validation_norm, X_test_norm = provider.normalize(X_train, X_validation, X_test)
    provider.save(X_train_df_mean, X_train_df_std)
    y_train_df_mean, y_train_df_std, y_train_norm, y_validation_norm, y_test_norm = provider.normalize(y_train, y_validation, y_test)


if __name__ == "__main__":
    data_parser = argparse.ArgumentParser()
    data_parser.add_argument("-dataset", type=str, required=True, help="Enter the dataset file name.")
    data_parser.add_argument("-train_ratio", type=float, required=False, default=0.6, help="Train set splitting")
    args = data_parser.parse_args()
    main()
