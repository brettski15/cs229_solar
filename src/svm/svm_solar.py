def main(train_set, train_labels, valid_set, valid_labels, test_set, test_labels):
    print("Running SVM main")

    print(f"Found {len(train_set.X)} examples with {len(train_set.labels)} labels.")

    train_data = train_set.X
    train_labels = train_set.labels
    train_area = train_set.get_area_labels()
    train_tiles = train_set.get_tile_count_labels()
    train_system = train_set.get_system_count_labels()

    # See above for how to access the data and labels


if __name__ == '__main__':
    print("Please use `python run.py --svm` to run this model")
