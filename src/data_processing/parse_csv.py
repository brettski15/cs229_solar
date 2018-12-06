import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

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


def split_df(data_matrix, label_matrix, train_pct=80, valid_pct=10):
    """
    A method for splitting the data and label pandas Dataframes into test, validation, and training sets.
    Raises a RuntimeError if the total training + validation percents exceed 100 (which would leave no test set)
    Prints a warning if the test set is less than 5% of the total data.

    :param data_matrix: The Pandas DataFrame of data
    :param label_matrix: The Pandas DataFrame of labels
    :param train_pct: The percent of data to use for training. Default: 80%
    :param valid_pct: The percent of data to use for validation. Default: 10%
    :return: A tuple of: training_data, training_labels, validation_data, validation_labels, test_data, test_labels
    """
    if 100 - train_pct - valid_pct <= 0:
        raise RuntimeError("Invalid data split--not test data available.")

    if 100 - train_pct - valid_pct < 5:
        print("!!!!!!!!!!!WARNING: Your test set is less than 5% of the available data. Is that too small?")

    m = len(data_matrix)

    train_count = int((train_pct / 100) * m)
    valid_count = int((valid_pct / 100) * m)

    frac_train = float(train_pct) / 100
    frac_valid = float(valid_pct) / 100
    frac_test = 1.0 - frac_train - frac_valid

    print(f"Putting {train_count} examples in the training set.")
    print(f"Putting {valid_count} examples in the validation set.")
    print(f"Leaving {len(data_matrix) - train_count - valid_count} examples in the test set.")

    train_data, x_test_valid, train_labels, label_train_valid = train_test_split(data_matrix, label_matrix,
                                                                                 test_size=1 - frac_train,
                                                                                 random_state=0)

    frac_test = frac_test / (frac_valid + frac_test)

    valid_data, test_data, valid_labels, test_labels = train_test_split(x_test_valid, label_train_valid,
                                                                        test_size=frac_test, random_state=0)

    # train_data = data_matrix[train_count]
    # train_labels = label_matrix[train_count]
    #
    # valid_data = data_matrix[train_count:train_count + valid_count]
    # valid_labels = label_matrix[train_count:train_count + valid_count]
    #
    # test_data = data_matrix[train_count + valid_count:]
    # test_labels = label_matrix[train_count + valid_count:]

    return train_data, train_labels, valid_data, valid_labels, test_data, test_labels


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


def get_df_from_csv(csv_path, partial_data=None):
    """
    Read 2 pandas DataFrames from the data. The first is all of the data (columns 5-183 plus an index column)
    The second is the set of label columns (2-4) plus an index column
    :param csv_path: The path to the csv file to read
    :param partial_data: [Optional] If set, only use a subset of the available examples
    :return:
    """
    if not os.path.isfile(csv_path):
        print(f"Cannot find the file at {csv_path}")
        raise RuntimeError(f"File {csv_path} not found")

    pd.set_option('use_inf_as_na', True)

    with open(csv_path, 'r', encoding='ISO-8859-1') as fp:
        data_cols = [i for i in range(2, 183)]

        d_matrix = pd.read_csv(fp, nrows=partial_data, usecols=data_cols, na_values=[''],
                               encoding='ISO-8859-1')
        # rename_col = d_matrix.columns.values[0]
        # d_matrix.rename(columns={rename_col: 'idx'}, inplace=True)
        proxy_label_cols = [
            'tile_count_res',
            'tile_count_nonres',
            'solar_system_count_res',
            'solar_system_count_nonres',
            'total_panel_area_res',
            'total_panel_area_nonres',
            'system_per_household',
            'log_system_per_household',
            'system_per_household_adj1',
            'log_system_per_household_adj1',
            'system_per_household_adj2',
            'log_system_per_household_adj2',
            'fips',
            'existing_installs_count',
            'number_of_panels_median',
            'number_of_panels_total',
            'area_per_area',
            'area_per_population',
            'fc',
            'ft'
        ]
        d_matrix = d_matrix.drop(proxy_label_cols, axis=1)
        string_cols = [
            'county',
            'state',
            'electricity_price_transportation'
        ]
        d_matrix = d_matrix.drop(string_cols, axis=1)

        d_matrix.replace([np.inf, -np.inf], np.nan)
        d_matrix.dropna(inplace=True)
        d_matrix = d_matrix.astype(float, errors='ignore')
        d_matrix = d_matrix.reset_index(drop=True)
        print(f"\033[91mAfter dropping rows with NaNs, {len(d_matrix.index)} rows remaining.\033[0m")
        # print(d_matrix)

        seed = 1992
        print(f"Shuffling data with seed {seed}")
        d_matrix = shuffle(d_matrix, random_state=seed)

        labels_matrix = d_matrix.ix[:, 1:4]
        print(labels_matrix.head())
        print(d_matrix.head())
        d_matrix = d_matrix.drop(['tile_count', 'solar_system_count', 'total_panel_area'], axis=1)
        # print(labels_matrix)

    for row in d_matrix.itertuples(index=True, name='Pandas'):
        for i in row:
            if i == np.Inf:
                print(f"Found string in row. Item: {i}")
                print(row)
                continue

    d_matrix = d_matrix.reset_index(drop=True)
    labels_matrix = labels_matrix.reset_index(drop=True)

    return d_matrix, labels_matrix


def get_examples_from_csv(csv_path, partial_data=0, ret_simple_matrix=False):
    """
    A method to parse the csv data. It assumes that the csv_file has 2 index columns and ditches one of them.
    It also renames the first column (after removal) to 'index'
    :param csv_path: The path to the csv file to pull data from
    :param partial_data: (Optional) The number of examples to use from the dataset
    :param ret_simple_matrix: (Optional) If true, return a simple matrix. Else return a SolarMatrix
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
                if num_examples >= partial_data > 0:
                    print(f"Opting to not use all data for efficiency. Stopping at {partial_data} examples.")
                    break

    if ret_simple_matrix:
        return SimpleMatrix(feature_matrix, labels)
    # else:
    #     return SolarMatrix(examples, labels, header)


if __name__ == '__main__':
    # get_examples_from_csv(DATA_FILE)
    get_df_from_csv(DATA_FILE)
