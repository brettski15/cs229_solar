import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from .neural_plotter import plot_history, plot_test
import pickle

# https://medium.com/@rajatgupta310198/getting-started-with-neural-network-for-regression-and-tensorflow-58ad3bd75223
# https://www.kaggle.com/zoupet/neural-network-model-for-house-prices-tensorflow
# https://www.tensorflow.org/tutorials/keras/basic_regression

def save_data(save_filename, save_obj):
    print('\nSaving data...')
    
    with open(save_filename, 'wb') as fi:
        pickle.dump(save_obj, fi)
        
    print('Save data complete. Object saved as:', save_filename)
    
def main(train_set, train_labels, valid_set, valid_labels, test_set, test_labels):
    print("Running NN main")

    # See above for how to access the data and labels
    
    # hyperparameters
    layer_dims = [256, 192, 160, 160, 160, 1]
    activations = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu']
    dropout = [0.1, 0.25, 0.1, 0.35, 0.50, 0]
    lr = 1e-4
    num_epochs = 100
    batch_size = 64
    
    save_path = 'z-model-and-weights.h5'
    
    # initialize X, Y
    X_train = train_set
    X_valid = valid_set
    X_test = test_set
    
    Y_train = train_labels
    Y_valid = valid_labels
    Y_test = test_labels
    
    # save_data('data.h5', [X_train, Y_train, X_valid, Y_valid, X_test, Y_test])

    # convert to np and slice X, Y
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = convert2np(X_train, Y_train, X_valid, Y_valid, X_test, Y_test)
    
    # shuffle
    X_train, Y_train = shuffle(X_train, Y_train)
    X_valid, Y_valid = shuffle(X_valid, Y_valid)
    
    # normalize features
    X_train, X_valid, X_test = norm_features(X_train, X_valid, X_test)
    
    # build model
    n_train = X_train.shape[1]
    model = build_model(layer_dims, activations, dropout, n_train, lr)
        
    # train model
    model, history = train_model(model, X_train, Y_train, X_valid, Y_valid, batch_size, num_epochs)
    model.save(save_path)
    
#    # plot train/valid loss (outputs graphs of MSE, MAE, and R2 Coeff.)
#    print_stats(history)
#    plot_history(history)
#    
#    # predict using model
#    predictions = model.predict(X_test, verbose = 0)
#        
#    # plot test results (outputs graphs of labels vs. predictions and histogram of prediction error)
#    plot_test(predictions, Y_test)

def convert2np(X_train, Y_train, X_valid, Y_valid, X_test, Y_test):
    
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_valid, Y_valid = np.array(X_valid), np.array(Y_valid)
    X_test, Y_test = np.array(X_test), np.array(Y_test)
    
    X_train = X_train[:, :]
    X_valid = X_valid[:, :]
    X_test = X_test[:, :]
    
    Y_train = np.reshape(Y_train[:, 0], (Y_train.shape[0], 1))
    Y_valid = np.reshape(Y_valid[:, 0], (Y_valid.shape[0], 1))
    Y_test = np.reshape(Y_test[:, 0], (Y_test.shape[0], 1))
    
    print ('\n---------------NN Parameters----------------')
    print ('Train features shape:', X_train.shape)
    print ('Train labels shape:', Y_train.shape)
    print ('Valid features shape:', X_valid.shape)
    print ('Valid labels shape:', Y_valid.shape)
    print ('Test features shape:', X_test.shape)
    print ('Test labels shape:', Y_test.shape)
    print ('--------------------------------------------\n')
    
    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test 

def shuffle(X, Y):
    
    order = np.argsort(np.random.random(X.shape[0]))
    X = X[order, :]
    Y = Y[order, :]
    
    return X, Y

def norm_features(X_train, X_valid, X_test):
    
    print ('\n---------------Normalizing Features----------------')
    
    mean = X_train.mean(axis = 0)
    std = X_train.std(axis = 0)
    
    X_train = (X_train - mean) / std
    X_valid = (X_valid - mean) / std
    X_test = (X_test - mean) / std

    print ('X_train, X_valid, X_test normalized to zero mean and STD 1.\n')

    return X_train, X_valid, X_test  

def build_model(layer_dims, activations, dropout, n_train, lr):
    
    print ('\n---------------Building NN Model----------------')
    
    tf.reset_default_graph()
    K.backend.clear_session()
    
    model = K.models.Sequential()
    
    model.add(K.layers.Dense(layer_dims[0], input_dim = n_train))
    model.add(K.layers.Activation(activations[0]))
    for i in range(1, len(layer_dims)):
        model.add(K.layers.Dense(layer_dims[i]))
        model.add(K.layers.Dropout(dropout[i]))
        model.add(K.layers.Activation(activations[i]))

    model.compile(loss = 'mse', optimizer = K.optimizers.Adam(lr), metrics = ['mae'])
#    model.compile(loss = 'mse', optimizer = K.optimizers.Adam(lr), metrics = ['mae', r2_keras])
    
    print ('\n------------Building NN Model Complete------------------')
    
    model.summary()
    
    return model
    
def train_model(model, X_train, Y_train, X_valid, Y_valid, batch_size, num_epochs):
    
    print ('\n---------------Training/Validating NN Model----------------')
    
    history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = num_epochs, verbose = 0, validation_data = (X_valid, Y_valid), shuffle = True, callbacks = [progress()])
    
    print ('\n-----------Training/Validating NN Model Complete----------------')
    return model, history

def print_stats(history):
    
    min_train_loss = min(history.history['loss'])
    min_val_loss = min(history.history['val_loss'])
    max_r2_train = max(history.history['r2_keras'])
    max_r2_val = max(history.history['val_r2_keras'])
    
    print('Min Train Loss:', '%.3f' % min_train_loss)
    print('Min Valid Loss:', '%.3f' % min_val_loss)
    print('Max Train R2 Coeff.:', '%.3f' % max_r2_train)
    print('Max Valid R2 Coeff.', '%.3f' % max_r2_val)
    
class progress(K.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 10 == 0: print('Current Epoch:', epoch)
    
def r2_keras(y_true, y_pred):
    SS_res =  K.backend.sum(K.backend.square(y_true - y_pred)) 
    SS_tot = K.backend.sum(K.backend.square(y_true - K.backend.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.backend.epsilon()) )
    
if __name__ == '__main__':
    print("Please use `python run.py --nn` to run this model")
