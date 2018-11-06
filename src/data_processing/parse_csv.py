import os
# from solar_common.solar_structures import SolarExample, SolarMatrix, SolarLabel, SimpleMatrix
from solar_common.solar_structures import SolarLabel, SimpleMatrix


DATA_FILE = "../data/tract_all.csv"


def split_simple_data(data_matrix, train_pct=60, valid_pct=20):
    if 100 - train_pct - valid_pct <= 0:
        raise RuntimeError("Invalid data split--not test data available.")

    if 100 - train_pct - valid_pct < 5:
        print("!!!!!!!!!!!WARNING: Your test set is less than 5% of the available data. Is that too small?")

    m = len(data_matrix.labels)

    train_count = int((train_pct / 100) * m)
    valid_count = int((valid_pct / 100) * m)

    train_matrix = data_matrix.X[0:train_count]
    train_labels = data_matrix.labels[0:train_count]
    train_set = SimpleMatrix(train_matrix, train_labels)

    valid_matrix = data_matrix.X[train_count:-valid_count]
    valid_labels = data_matrix.labels[train_count:-valid_count]
    valid_set = SimpleMatrix(valid_matrix, valid_labels)

    test_matrix = data_matrix.X[-valid_count:]
    test_labels = data_matrix.labels[-valid_count:]
    test_set = SimpleMatrix(test_matrix, test_labels)

    return train_set, valid_set, test_set


# def split_data(data_matrix, train_pct=60, valid_pct=20):
#     if 100 - train_pct - valid_pct <= 0:
#         raise RuntimeError("Invalid data split--not test data available.")
#
#     if 100 - train_pct - valid_pct < 5:
#         print("!!!!!!!!!!!WARNING: Your test set is less than 5% of the available data. Is that too small?")
#
#     examples = data_matrix.data
#     labels = data_matrix.labels
#
#     m = len(examples)
#
#     train_count = int((train_pct / 100) * m)
#     valid_count = int((valid_pct / 100) * m)
#     # test_count = m - train_count - valid_count
#
#     train_matrix = SolarMatrix(examples[0:train_count], labels[0:train_count],
#                                data_matrix.headers)
#
#     valid_matrix = SolarMatrix(examples[train_count:-valid_count], labels[train_count:-valid_count],
#                                data_matrix.headers)
#
#     test_matrix = SolarMatrix(examples[-valid_count:], labels[-valid_count:],
#                           data_matrix.headers)
#
#     return train_matrix, valid_matrix, test_matrix


def get_examples_from_csv(csv_path, partial_data=0, ret_simple_matrix=False):
    """
    A method to parse the csv data. It assumes that the csv_file has 2 index columns and ditches one of them.
    It also renames the first column (after removal) to 'index'
    :param csv_path: The path to the csv file to pull data from
    :param partial_data: (Optional) The number of examples to use from the dataset
    :return: SolarMatrix -- a structure containing all of the header names as well as all examples
    """
    if not os.path.isfile(csv_path):
        print(f"Cannot find the file at {csv_path}")
        raise RuntimeError(f"File {csv_path} not found")

    # examples = []
    labels = []
    num_examples = 0
    header = None
    feature_matrix = []

    with open(csv_path, 'r', encoding="ISO-8859-1") as fp:
        for l in fp:
            if not header:
                header = l.split(',')[4:]
                header[0] = 'index'
                # print(header)
            else:
                items = l.split(',')[1:]
                # print(items)
                label = SolarLabel(items[3], items[2], items[1])
                ex_vals = items[4:]
                feature_matrix.append(ex_vals)
                # ex = SolarExample(ex_vals, label)
                # examples.append(ex)
                labels.append(label)
                num_examples += 1
                if partial_data > 0 and num_examples >= partial_data:
                    print(f"Opting to not use all data for efficiency. Stopping at {partial_data} examples.")
                    break

    if ret_simple_matrix:
        return SimpleMatrix(feature_matrix, labels)
    # else:
    #     return SolarMatrix(examples, labels, header)


if __name__ == '__main__':
    get_examples_from_csv(DATA_FILE)
