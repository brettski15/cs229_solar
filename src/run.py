import argparse
from neural_network.neural_solar import main as nn_main
from svm.svm_solar import main as svm_main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--svm', action='store_true', help="If specified, run the SVM implementation")
    parser.add_argument('--nn', action='store_true', help="If specified, run the Neural Network implementation")
    args = parser.parse_args()

    if args.nn:
        nn_main()

    if args.svm:
        svm_main()


if __name__ == '__main__':
    main()