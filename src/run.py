import argparse
import numpy as np

from neural_network.neural_solar import main as nn_main
from svm.svm_solar import main as svm_main
from data_processing.parse_csv import get_examples_from_csv, split_simple_data


DATA_PATH = "../data/tract_all.csv"
TOTAL_DATA_SIZE = 72538
RESERVE_TEST_DATA = 14500


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--svm', action='store_true', help="If specified, run the SVM implementation")
    parser.add_argument('--nn', action='store_true', help="If specified, run the Neural Network implementation")
    parser.add_argument('--count', '-c', type=int, default=1000,
                        help="The number of pieces of data to use. 0 for all data")
    args = parser.parse_args()

    print(f"Pulling {args.count} examples from the CSV")

    if args.count > TOTAL_DATA_SIZE - RESERVE_TEST_DATA or args.count <= 0:
        data_count = TOTAL_DATA_SIZE - RESERVE_TEST_DATA
        print(f"You have requested {args.count} rows of data, which would not leave {RESERVE_TEST_DATA} untouched rows "
              f"to be used for production test data. Reducing your requested data size to {data_count}.")
    else:
        data_count = args.count

    print(f"Requesting {data_count} rows of data.")

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
