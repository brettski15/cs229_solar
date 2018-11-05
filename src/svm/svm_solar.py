from data_processing.parse_csv import get_examples_from_csv


DATA_PATH = "../data/tract_all.csv"


def main():
    print("Running SVM main")
    training_set = get_examples_from_csv(DATA_PATH, 1000)
    print(f"Parsed CSV and found {len(training_set.data)} examples with {len(training_set.labels)} "
          f"and {len(training_set.headers)} features.")
    # See above for how to access the data, labels, and (if desired) headers


if __name__ == '__main__':
    main()