import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from neural_network.neural_solar import main as nn_main
from svm.svm_solar import main as svm_main
from data_processing.parse_csv import get_examples_from_csv, split_simple_data, get_df_from_csv, split_df


DATA_PATH = "../data/tract_all.csv"
TOTAL_DATA_SIZE = 72538
RESERVE_TEST_DATA = 14500


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--svm', action='store_true', help="If specified, run the SVM implementation.")
    parser.add_argument('--nn', action='store_true', help="If specified, run the Neural Network implementation.")
    parser.add_argument('--count', '-c', type=int, default=1000,
                        help="The number of pieces of data to use. 0 for all data.")
    parser.add_argument('--df', action='store_true', help="If specified, pull data into a Pandas DataFrame.")
    parser.add_argument('--pca', action='store_true', help="If specified, run PCA analysis. Requires --df as well.")

    args = parser.parse_args()

    print(f"Pulling {args.count} examples from the CSV")

    if args.count > TOTAL_DATA_SIZE - RESERVE_TEST_DATA or args.count <= 0:
        data_count = TOTAL_DATA_SIZE - RESERVE_TEST_DATA
        print(f"You have requested {args.count} rows of data, which would not leave {RESERVE_TEST_DATA} untouched rows "
              f"to be used for production test data. Reducing your requested data size to {data_count}.")
    else:
        data_count = args.count

    print(f"Requesting {data_count} rows of data.")

    if args.df:
        data, labels = get_df_from_csv(DATA_PATH, data_count)
        # print(list(data.columns.values))
        train_set, train_labels, valid_set, valid_labels, test_set, test_labels = split_df(data, labels)
        # print(train_set.head())

        if args.pca:
            print("Scaling data")
            sc = StandardScaler()
            x_train = sc.fit_transform(train_set)
            x_test = sc.transform(test_set)

            print("Applying PCA")
            pca = PCA()
            x_train = pca.fit_transform(x_train)
            x_test = pca.transform(x_test)
            print(pca.explained_variance_ratio_)
    else:
        full_data = get_examples_from_csv(DATA_PATH, data_count, ret_simple_matrix=True)
        train_set, valid_set, test_set = split_simple_data(full_data)

        # train_set, valid_set, test_set = split_data(full_data, train_pct=60, valid_pct=20)

        print(f"Train size: {train_set.labels.shape[0]}. "
              f"Valid size: {valid_set.labels.shape[0]}. "
              f"Test size: {test_set.labels.shape[0]}")

        if args.nn:
            nn_main(train_set, valid_set, test_set)

        if args.svm:
            svm_main(train_set, valid_set, test_set)


if __name__ == '__main__':
    main()
