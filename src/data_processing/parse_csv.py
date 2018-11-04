import os
from solar_common.solar_structures import SolarExample, SolarMatrix, SolarLabel


DATA_FILE = "../data/tract_all.csv"


def get_examples_from_csv(csv_path):
    if not os.path.isfile(csv_path):
        print(f"Cannot find the file at {csv_path}")
        raise RuntimeError(f"File {csv_path} not found")

    examples = []
    num_examples = 0
    header = None

    with open(csv_path, 'r') as fp:
        for l in fp:
            if not header:
                header = l.split(',')
            else:
                items = l.split(',')
                print(items)
                label = SolarLabel(items[3], items[2], items[1])
                ex_vals = items[4:]
                ex = SolarExample(ex_vals, label)
                examples.append(ex)
                num_examples += 1
                print(label)
                break

    matrix = SolarMatrix(examples, header)
    return matrix


if __name__ == '__main__':
    get_examples_from_csv(DATA_FILE)
