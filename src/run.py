import argparse

from neural_network.neural_solar import main as nn_main
from svm.svm_solar import main as svm_main
from data_processing.parse_csv import get_examples_from_csv, split_simple_data, get_df_from_csv, split_df
from data_processing.pca import  pca_main


DATA_PATH = "../data/tract_all.csv"
TOTAL_DATA_SIZE = 72538
RESERVE_TEST_DATA = 14500


# def shuffle(features, labels):
#     np.random.seed(1992)
#
#     arg_order = np.random.random(labels.shape[0])
#     order = np.argsort(arg_order)
#
#     features_shuffled = features[order]
#     labels_shuffled = np.reshape(labels[order], (labels.shape[0], 1))
#
#     return features_shuffled, labels_shuffled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--svm', action='store_true', help="If specified, run the SVM implementation.")
    parser.add_argument('--nn', action='store_true', help="If specified, run the Neural Network implementation.")
    parser.add_argument('--count', '-c', type=int, default=1000,
                        help="The number of pieces of data to use. 0 for all data.")
    parser.add_argument('--df', action='store_true', help="If specified, pull data into a Pandas DataFrame.")
    parser.add_argument('--pca', action='store_true', help="If specified, run PCA analysis. Requires --df as well.")
    parser.add_argument('--heatmap', action='store_true',
                        help="If specified, run a plotly choropleth for data density analysis.")

    args = parser.parse_args()

    print(f"Pulling {args.count} examples from the CSV")

    if args.count > TOTAL_DATA_SIZE or args.count <= 0:
        data_count = TOTAL_DATA_SIZE
        print(f"Pulling ALL {TOTAL_DATA_SIZE} examples from the data set.")
    else:
        data_count = args.count

    print(f"Requesting {data_count} rows of data.")

    data, labels = get_df_from_csv(DATA_PATH, data_count, args.heatmap)
    # data.to_csv('data.csv')
    # print(list(data.columns.values))
    train_set, train_labels, valid_set, valid_labels, test_set, test_labels = split_df(data, labels)
    # print(train_set.head())

    if args.pca:
        pca_main(train_set, train_labels, test_set, test_labels)

    if args.nn:
        nn_main(train_set, train_labels, valid_set, valid_labels, test_set, test_labels)

    if args.svm:
        svm_main(train_set, train_labels, valid_set, valid_labels, test_set, test_labels)


if __name__ == '__main__':
    main()
