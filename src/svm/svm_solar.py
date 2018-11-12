
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

def main(train_set, valid_set, test_set):
    # #############################################################################
    # Generate sample data
    #X = np.sort(5 * np.random.rand(40, 1), axis=0)
    #y = np.sin(X).ravel()

    # #############################################################################
    # Add noise to targets
    #y[::5] += 3 * (0.5 - np.random.rand(8))

    # #############################################################################
    # Fit regression model
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    y_rbf = svr_rbf.fit(train_set, train_labels).predict(train_set)
    y_lin = svr_lin.fit(train_set, train_labels).predict(train_set)
    y_poly = svr_poly.fit(train_set, train_labels).predict(train_set)

    # #############################################################################
    # Look at the results
    lw = 2
    plt.scatter(train_set, train_labels, color='darkorange', label='data')
    plt.plot(train_set, y_rbf, color='navy', lw=lw, label='RBF model')
    plt.plot(train_set, y_lin, color='c', lw=lw, label='Linear model')
    plt.plot(train_set, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()


    print(f"Found {len(train_set.X)} examples with {len(train_set.labels)} labels.")

    train_data = train_set.X
    train_labels = train_set.labels
    train_area = train_set.get_area_labels()
    train_tiles = train_set.get_tile_count_labels()
    train_system = train_set.get_system_count_labels()

    # See above for how to access the data and labels


 if __name__ == '__main__':
     print("Please use `python run.py --svm` to run this model")
