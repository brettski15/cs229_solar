
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

def main(train_set, train_labels, valid_set, valid_labels, test_set, test_labels):

    # #############################################################################
    # Generate sample data
    #X_train = np.sort(5 * np.random.rand(40, 1), axis=0)
    #y_train = np.sin(X).ravel()

    # #############################################################################
    # Add noise to targets
    #y_train[::5] += 3 * (0.5 - np.random.rand(8))

    X_train = train_set
    X_valid = valid_set
    X_test = test_set

    Y_train = train_labels
    Y_valid = valid_labels
    Y_test = test_labels

    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = convert2np(X_train, Y_train, X_valid, Y_valid, X_test, Y_test)
    X_train, Y_train = shuffle(X_train, Y_train)
    X_valid, Y_valid = shuffle(X_valid, Y_valid)
    X_train, X_valid, X_test = norm_features(X_train, X_valid, X_test)

    Y_train = Y_train.T
    Y_train = Y_train.ravel()
    Y_valid = Y_valid.T
    Y_valid = Y_valid.ravel()
    Y_test = Y_test.T
    Y_test = Y_test.ravel()

    # #############################################################################
    #SVR using RBF
    print("Fitting SVR using rbf... \n")
    svr_rbf = SVR(C=1e3, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='scale', kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
    svr_rbf.fit(X_train, Y_train)
    confidence_rbf_validation = svr_rbf.score(X_valid,Y_valid)
    confidence_rbf_test = svr_rbf.score(X_test,Y_test)
    print("Average confidence for validation data using rbf kernel is %f" %confidence_rbf_validation)
    print("Average confidence for test data using rbf kernel is %f" %confidence_rbf_test)
    print("\n")
    # #############################################################################
    # SVR using Linear
    print("Fitting SVR using Linear... \n")
    svr_linear = SVR(C=1e3, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='scale', kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
    svr_linear.fit(X_train, Y_train)
    confidence_linear_validation = svr_linear.score(X_valid,Y_valid)
    confidence_linear_test = svr_linear.score(X_test,Y_test)
    print("Average confidence for validation data using rbf kernel is %f" %confidence_linear_validation)
    print("Average confidence for test data using rbf kernel is %f" %confidence_linear_test)
    print("\n")
    # #############################################################################
    # SVR using Poly
    print("Fitting SVR using Poly... \n")
    svr_poly = SVR(C=1e3, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='scale', kernel='poly', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
    svr_poly.fit(X_train, Y_train)
    confidence_poly_validation = svr_poly.score(X_valid,Y_valid)
    confidence_poly_test = svr_poly.score(X_test,Y_test)
    print("Average confidence for validation data using rbf kernel is %f" %confidence_poly_validation)
    print("Average confidence for test data using rbf kernel is %f" %confidence_poly_test)
    print("\n")
    # #############################################################################


    Y_rbf_test_pridict = svr_rbf.predict(X_test)
    Y_linear_test_pridict = svr_linear.predict(X_test)
    Y_poly_test_pridict = svr_poly.predict(X_test)
    print("Y_rbf_test_pridict\n")
    print(Y_rbf_test_pridict)
    print("=====================================================================")
    print("Y_linear_test_pridict\n")
    print(Y_linear_test_pridict)
    print("=====================================================================")
    print("Y_poly_test_pridict\n")
    print(Y_poly_test_pridict)
    print("=====================================================================")
    print("Y_test\n")
    print(Y_test)
    print("=====================================================================")

    # #############################################################################
    # Visualize the prediction result

    lw = 2
    #X_Axis = np.arange(0,76)
    #X_Axis = np.array([X_Axis])
    #X_Axis = X_Axis.T
    #plt.scatter(X_Axis, Y_test, color='black', label='Y_Test_Data')
    plt.xlabel("Actual Label")
    plt.ylabel("Predict Label")
    # #############################################################################
    # Plot RBF result
    #plt.scatter(Y_test, Y_rbf_test_pridict, color='blue', lw=lw, label='RBF Kernel')
    #plt.title("SVR using RBF Kernel \n Average confidence: %f" %confidence_rbf_test)

    # #############################################################################
    # Plot Linear result
    #plt.scatter(Y_test, Y_linear_test_pridict, color='red', lw=lw, label='Linear Kernel')
    #plt.title("SVR using Linear Kernel \n Average confidence: %f" % confidence_linear_test)

    # #############################################################################
    # Plot Poly result
    plt.scatter(Y_test, Y_poly_test_pridict, color='green', lw=lw, label='Polynomial Kernel')
    plt.title("SVR using Polynomial Kernel \n Average confidence: %f" % confidence_poly_test)

    # #############################################################################

    plt.legend()
    plt.show()

    #print(f"Found {len(X.X)} examples with {len(X.labels)} labels.")

    #train_data = X.X
    #y = X.labels
    #train_area = X.get_area_labels()
    #train_tiles = X.get_tile_count_labels()
    #train_system = X.get_system_count_labels()







# #############################################################################
# Some excellent data manipulation methods
# Author: Eddie Sun

def convert2np(X_train, Y_train, X_valid, Y_valid, X_test, Y_test):
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_valid, Y_valid = np.array(X_valid), np.array(Y_valid)
    X_test, Y_test = np.array(X_test), np.array(Y_test)

    X_train = X_train[:, 24:150]
    X_valid = X_valid[:, 24:150]
    X_test = X_test[:, 24:150]

    Y_train = np.reshape(Y_train[:, 0], (Y_train.shape[0], 1))
    Y_valid = np.reshape(Y_valid[:, 0], (Y_valid.shape[0], 1))
    Y_test = np.reshape(Y_test[:, 0], (Y_test.shape[0], 1))

    print('\n---------------SVR Input Shape----------------')
    print('Train features shape:', X_train.shape)
    print('Train labels shape:', Y_train.shape)
    print('Valid features shape:', X_valid.shape)
    print('Valid labels shape:', Y_valid.shape)
    print('Test features shape:', X_test.shape)
    print('Test labels shape:', Y_test.shape)
    print('--------------------------------------------\n')

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test

# #############################################################################
# Some excellent data manipulation methods
# Author: Eddie Sun

def shuffle(X, Y):
    order = np.argsort(np.random.random(X.shape[0]))
    X = X[order, :]
    Y = Y[order, :]

    return X, Y

# #############################################################################
# Some excellent data manipulation methods
# Author: Eddie Sun

def norm_features(X_train, X_valid, X_test):
    print('\n---------------Normalizing Features----------------')

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    X_train = (X_train - mean) / std
    X_valid = (X_valid - mean) / std
    X_test = (X_test - mean) / std

    print('X_train, X_valid, X_test normalized to zero mean and STD 1.\n')

    return X_train, X_valid, X_test

# See above for how to access the data and labels

if __name__ == '__main__':
     print("Please use `python run.py --svm` to run this model")
