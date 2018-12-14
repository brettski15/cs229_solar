import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt


def main(train_set, train_labels, valid_set, valid_labels, test_set, test_labels):

    # #############################################################################
    # Generate sample data
    # x_train = np.sort(5 * np.random.rand(40, 1), axis=0)
    # y_train = np.sin(X).ravel()

    # #############################################################################
    # Add noise to targets
    # y_train[::5] += 3 * (0.5 - np.random.rand(8))

    x_train = train_set
    x_valid = valid_set
    x_test = test_set

    y_train = train_labels
    y_valid = valid_labels
    y_test = test_labels

    x_train, y_train, x_valid, y_valid, x_test, y_test = convert2np(x_train, y_train, x_valid, y_valid, x_test, y_test)
    x_train, y_train = shuffle(x_train, y_train)
    x_valid, y_valid = shuffle(x_valid, y_valid)
    x_train, x_valid, x_test = norm_features(x_train, x_valid, x_test)

    y_train = y_train.T
    y_train = y_train.ravel()
    y_valid = y_valid.T
    y_valid = y_valid.ravel()
    y_test = y_test.T
    y_test = y_test.ravel()

    # #############################################################################
    # SVR using RBF
    print("Fitting SVR using rbf... \n")
    svr_rbf = SVR(C=1e3, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='scale', kernel='rbf', max_iter=-1,
                  shrinking=True, tol=0.001, verbose=False)
    svr_rbf.fit(x_train, y_train)
    confidence_rbf_validation = svr_rbf.score(x_valid, y_valid)
    confidence_rbf_test = svr_rbf.score(x_test, y_test)
    print("Average confidence for validation data using rbf kernel is %f" % confidence_rbf_validation)
    print("Average confidence for test data using rbf kernel is %f" % confidence_rbf_test)
    print("\n")
    # #############################################################################
    # SVR using Linear
    print("Fitting SVR using Linear... \n")
    svr_linear = SVR(C=1e3, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='scale', kernel='linear',
                     max_iter=-1, shrinking=True, tol=0.001, verbose=False)
    svr_linear.fit(x_train, y_train)
    confidence_linear_validation = svr_linear.score(x_valid, y_valid)
    confidence_linear_test = svr_linear.score(x_test, y_test)
    print("Average confidence for validation data using rbf kernel is %f" % confidence_linear_validation)
    print("Average confidence for test data using rbf kernel is %f" % confidence_linear_test)
    print("\n")
    # #############################################################################
    # SVR using Poly
    print("Fitting SVR using Poly... \n")
    svr_poly = SVR(C=1e3, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='scale', kernel='poly', max_iter=-1,
                   shrinking=True, tol=0.001, verbose=False)
    svr_poly.fit(x_train, y_train)
    confidence_poly_validation = svr_poly.score(x_valid, y_valid)
    confidence_poly_test = svr_poly.score(x_test, y_test)
    print("Average confidence for validation data using rbf kernel is %f" % confidence_poly_validation)
    print("Average confidence for test data using rbf kernel is %f" % confidence_poly_test)
    print("\n")
    # #############################################################################

    y_rbf_test_predict = svr_rbf.predict(x_test)
    y_linear_test_predict = svr_linear.predict(x_test)
    y_poly_test_predict = svr_poly.predict(x_test)
    print("y_rbf_test_predict\n")
    print(y_rbf_test_predict)
    print("=====================================================================")
    print("y_linear_test_predict\n")
    print(y_linear_test_predict)
    print("=====================================================================")
    print("y_poly_test_predict\n")
    print(y_poly_test_predict)
    print("=====================================================================")
    print("y_test\n")
    print(y_test)
    print("=====================================================================")

    # #############################################################################
    # Visualize the prediction result

    lw = 2
    # X_Axis = np.arange(0,76)
    # X_Axis = np.array([X_Axis])
    # X_Axis = X_Axis.T
    # plt.scatter(X_Axis, y_test, color='black', label='Y_Test_Data')
    plt.xlabel("Actual Label")
    plt.ylabel("Predict Label")
    # #############################################################################
    # Plot RBF result
    # plt.scatter(y_test, y_rbf_test_predict, color='blue', lw=lw, label='RBF Kernel')
    # plt.title("SVR using RBF Kernel \n Average confidence: %f" %confidence_rbf_test)

    # #############################################################################
    # Plot Linear result
    # plt.scatter(y_test, y_linear_test_predict, color='red', lw=lw, label='Linear Kernel')
    # plt.title("SVR using Linear Kernel \n Average confidence: %f" % confidence_linear_test)

    # #############################################################################
    # Plot Poly result
    plt.scatter(y_test, y_poly_test_predict, color='green', lw=lw, label='Polynomial Kernel')
    plt.title("SVR using Polynomial Kernel \n Average confidence: %f" % confidence_poly_test)

    # #############################################################################

    plt.legend()
    plt.show()

    # print(f"Found {len(X.X)} examples with {len(X.labels)} labels.")

    # train_data = X.X
    # y = X.labels
    # train_area = X.get_area_labels()
    # train_tiles = X.get_tile_count_labels()
    # train_system = X.get_system_count_labels()


# #############################################################################
# Some excellent data manipulation methods
# Author: Eddie Sun

def convert2np(x_train, y_train, x_valid, y_valid, x_test, y_test):
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_valid, y_valid = np.array(x_valid), np.array(y_valid)
    x_test, y_test = np.array(x_test), np.array(y_test)

    x_train = x_train[:, 24:150]
    x_valid = x_valid[:, 24:150]
    x_test = x_test[:, 24:150]

    y_train = np.reshape(y_train[:, 0], (y_train.shape[0], 1))
    y_valid = np.reshape(y_valid[:, 0], (y_valid.shape[0], 1))
    y_test = np.reshape(y_test[:, 0], (y_test.shape[0], 1))

    print('\n---------------SVR Input Shape----------------')
    print('Train features shape:', x_train.shape)
    print('Train labels shape:', y_train.shape)
    print('Valid features shape:', x_valid.shape)
    print('Valid labels shape:', y_valid.shape)
    print('Test features shape:', x_test.shape)
    print('Test labels shape:', y_test.shape)
    print('--------------------------------------------\n')

    return x_train, y_train, x_valid, y_valid, x_test, y_test


# #############################################################################
# Some excellent data manipulation methods
# Author: Eddie Sun

def shuffle(x, y):
    order = np.argsort(np.random.random(x.shape[0]))
    x = x[order, :]
    y = y[order, :]

    return x, y


# #############################################################################
# Some excellent data manipulation methods
# Author: Eddie Sun

def norm_features(x_train, x_valid, x_test):
    print('\n---------------Normalizing Features----------------')

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)

    x_train = (x_train - mean) / std
    x_valid = (x_valid - mean) / std
    x_test = (x_test - mean) / std

    print('x_train, X_valid, X_test normalized to zero mean and STD 1.\n')

    return x_train, x_valid, x_test

# See above for how to access the data and labels


if __name__ == '__main__':
    print("Please use `python run.py --svm` to run this model")
