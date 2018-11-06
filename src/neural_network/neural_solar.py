def main(train_set, valid_set, test_set):
    print("Running NN main")

    print(f"Found {len(train_set.X)} examples with {len(train_set.labels)} labels.")

    train_data = train_set.X
    train_labels = train_set.labels
    train_area = train_set.get_area_labels()
    train_tiles = train_set.get_tile_count_labels()
    train_system = train_set.get_system_count_labels()
    # See above for how to access the data, labels, and (if desired) headers


if __name__ == '__main__':
    print("Please use `python run.py --nn` to run this model")
