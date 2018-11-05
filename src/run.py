import argparse
from neural_network.neural_solar import main as nn_main
from svm.svm_solar import main as svm_main
from data_processing.parse_csv import get_examples_from_csv, split_data


DATA_PATH = "../data/tract_all.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--svm', action='store_true', help="If specified, run the SVM implementation")
    parser.add_argument('--nn', action='store_true', help="If specified, run the Neural Network implementation")
    parser.add_argument('--count', '-c', type=int, default=1000, help="The number of pieces of data to use.")
    args = parser.parse_args()

    print(f"Pulling {args.count} examples from the CSV")

    full_data = get_examples_from_csv(DATA_PATH, args.count)

    train_set, valid_set, test_set = split_data(full_data, train_pct=60, valid_pct=20)
    return

    if args.nn:
        nn_main(train_set, valid_set, test_set)

    if args.svm:
        svm_main(train_set, valid_set, test_set)


if __name__ == '__main__':
    main()