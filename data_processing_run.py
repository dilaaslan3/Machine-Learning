import argparse
from Data_Provider import DataProvider
import os


def main():

    path = f"Output{os.sep}{args.out_folder}"
    if not os.path.exists(path):
        os.makedirs(path)
    provider = DataProvider(args.dataset, path)
    df = provider.read()
    X = provider.one_hot_encoding(df)
    X_train, X_validation, X_test, y_train, y_validation, y_test = provider.split(X, args.train_ratio, args.target)
    X_train_norm, X_validation_norm, X_test_norm = provider.normalize(X_train, X_validation, X_test, True)
    y_train_norm, y_validation_norm, y_test_norm = provider.normalize(y_train, y_validation, y_test, False)


if __name__ == "__main__":
    data_parser = argparse.ArgumentParser()
    data_parser.add_argument("-dataset", type=str, required=True, help="Enter the dataset file name.")
    data_parser.add_argument("-target", type=str, required=True, help="Enter the target name.")
    data_parser.add_argument("-train_ratio", type=float, required=False, default=0.6, help="Train set splitting")
    data_parser.add_argument("-out_folder", type=str, required=True, help="Output folder")
    args = data_parser.parse_args()
    main()
