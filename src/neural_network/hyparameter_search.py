import numpy as np
import tensorflow as tf
from tensorflow import keras as K
import matplotlib.pyplot as plt
import pickle
import pandas as pd

def main():
    print("Running NN main")

    # See above for how to access the data and labels
    
    hyps = pd.read_csv('hyperparameter_search.csv')
#    print (hyps)
    hyps = np.array(hyps)
    
    learning_rate = hyps[0, :]
    batch_size = hyps[1, :]
    num_layers = hyps[2, :]
    
    layer_dims = []
    dropout = []

    
    for i in range(hyps.shape[1]):
        layer_dims.append([])
        dropout.append([])
        
    for i in range(hyps.shape[1]):
        layer_dims[i] = hyps[3:9, i]
        dropout[i] = hyps[9:15, i]
        
    learning_rate_h, batch_size_h, num_layers_h, layer_dims_h, dropout_h = (np.array(learning_rate), np.array(batch_size),
                                                                            np.array(num_layers), np.array(layer_dims), np.array(dropout))
    stats = np.zeros((4, hyps.shape[1]))
    
    # initialize X, Y
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = open_data('data.h5')

    # convert to np and slice X, Y
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = convert2np(X_train, Y_train, X_valid, Y_valid, X_test, Y_test)
    
    # shuffle
    X_train, Y_train = shuffle(X_train, Y_train)
    X_valid, Y_valid = shuffle(X_valid, Y_valid)
    
    # normalize features
    X_train, X_valid, X_test = norm_features(X_train, X_valid, X_test)

    # hyperparameters
    
    activations = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu']
    num_epochs = 150
    
    for i in range(hyps.shape[1]):
        lr = learning_rate_h[i]
        batch_size = int(batch_size_h[i])
        num_layers = num_layers_h[i]
        layer_dims = layer_dims_h[i, :]
        dropout = dropout_h[i, :]
    
        # build model
        n_train = X_train.shape[1]
        model = build_model(layer_dims, activations, dropout, n_train, lr, num_layers)
            
        # train model
        model, history = train_model(model, X_train, Y_train, X_valid, Y_valid, batch_size, num_epochs)
        
        # plot train/valid loss (outputs graphs of MSE, MAE, and R2 Coeff.)
        print_stats(history)
        stats[0, i] = min(history.history['loss'])
        stats[1, i] = min(history.history['val_loss'])
        stats[2, i] = max(history.history['r2_keras'])
        stats[3, i] = max(history.history['val_r2_keras'])

        epoch = history.epoch
        train_r2 = np.array(history.history['r2_keras'])
        val_r2 = np.array(history.history['val_r2_keras'])
        
        plt.figure(2, figsize=(3.75, 2.75))
        plt.clf()
        plt.plot(epoch, train_r2, label = 'Train R2')
        plt.plot(epoch, val_r2, label = 'Valid R2')
        plt.legend()
        plt.ylim([0.6, 1])
        plt.xlabel('Epoch')
        plt.ylabel('R2 Coeff')
        plt.savefig('hyp_search_' + str(i) + '.png', bbox_inches="tight")
        del model
    
    np.savetxt('stats.csv', stats)

def open_data(save_filename):
    print('\nOpening', save_filename, '...')
    
    with open(save_filename, 'rb') as fi:
        load_temp = pickle.load(fi)
    loaded = load_temp      

    print('Open data complete.')
    
    return loaded

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
    
    np.random.seed(0)
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

def build_model(layer_dims, activations, dropout, n_train, lr, num_layers):
    
    print ('\n---------------Building NN Model----------------')
    
    tf.reset_default_graph()
    K.backend.clear_session()
    
    model = K.models.Sequential()
    
    model.add(K.layers.Dense(layer_dims[0], input_dim = n_train))
    model.add(K.layers.Dropout(dropout[0]))
    model.add(K.layers.Activation(activations[0]))
    for i in range(1, int(num_layers)):
        model.add(K.layers.Dense(layer_dims[i]))
        model.add(K.layers.Dropout(dropout[i]))
        model.add(K.layers.Activation(activations[i]))
    model.add(K.layers.Dense(1))
    model.add(K.layers.Activation('relu'))

#    model.compile(loss = 'mse', optimizer = K.optimizers.Adam(lr), metrics = ['mae'])
    model.compile(loss = 'mse', optimizer = K.optimizers.Adam(lr), metrics = ['mae', r2_keras])
    
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
    main()