import os
from solar_common.solar_structures import SolarExample, SolarMatrix, SolarLabel


DATA_FILE = "../data/tract_all.csv"


def get_examples_from_csv(csv_path, partial_data=0):
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

    examples = []
    labels = []
    num_examples = 0
    header = None

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
                ex = SolarExample(ex_vals, label)
                examples.append(ex)
                labels.append(label)
                num_examples += 1
                if partial_data > 0 and num_examples >= partial_data:
                    print(f"Opting to not use all data for efficiency. Stopping at {partial_data} examples.")
                    break

    matrix = SolarMatrix(examples, labels, header)
    print(matrix)
    return matrix


if __name__ == '__main__':
    get_examples_from_csv(DATA_FILE)
