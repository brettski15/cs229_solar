def main(train_set, valid_set, test_set):
    print("Running NN main")

    print(f"Found {len(train_set.data)} examples with {len(train_set.labels)} "
          f"and {len(train_set.headers)} features.")
    # See above for how to access the data, labels, and (if desired) headers


if __name__ == '__main__':
    print("Please use `python run.py --nn` to run this model")
