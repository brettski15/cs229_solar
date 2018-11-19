import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras as K

# https://medium.com/@rajatgupta310198/getting-started-with-neural-network-for-regression-and-tensorflow-58ad3bd75223
# https://www.kaggle.com/zoupet/neural-network-model-for-house-prices-tensorflow
# https://www.tensorflow.org/tutorials/keras/basic_regression

def main(train_set, train_labels, valid_set, valid_labels, test_set, test_labels):
    print("Running NN main")

    # See above for how to access the data and labels
    
    # hyperparameters
    layer_dims = [64, 64, 1]
    activations = ['relu', 'relu', 'relu']
    lr = 0.001
    optimizer = tf.train.RMSPropOptimizer(lr)
    num_epochs = 100
    batch_size = 64
    
    # initialize X, Y
    X_train = train_set
    X_valid = valid_set
    X_test = test_set
    
    Y_train = train_labels
    Y_valid = valid_labels
    Y_test = test_labels

    # convert to np and slice X, Y
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = convert2np(X_train, Y_train, X_valid, Y_valid, X_test, Y_test)
    
    # shuffle
    X_train, Y_train = shuffle(X_train, Y_train)
    X_valid, Y_valid = shuffle(X_valid, Y_valid)
    
    # normalize features
    X_train, X_valid, X_test = norm_features(X_train, X_valid, X_test)
    
    # build model
    n_train = X_train.shape[1]
    model = build_model(layer_dims, activations, optimizer, n_train)
        
    # train model
    model, history = train_model(model, X_train, Y_train, X_valid, Y_valid, batch_size, num_epochs)
    
    # plot train/valid loss (outputs graphs of MSE, MAE, and R2 Coeff.)
    plot_history(history)
    
    # predict using model
    predictions = model.predict(X_test, verbose = 0)
        
    # plot test results (outputs graphs of labels vs. predictions and histogram of prediction error)
    plot_test(predictions, Y_test)

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

def build_model(layer_dims, activations, optimizer, n_train):
    
    print ('\n---------------Building NN Model----------------')
    
    tf.reset_default_graph()
    K.backend.clear_session()
    
    model = K.models.Sequential()
    
    model.add(K.layers.Dense(layer_dims[0], input_dim = n_train))
    model.add(K.layers.Activation(activations[0]))
    for i in range(1, len(layer_dims)):
        model.add(K.layers.Dense(layer_dims[i]))
        model.add(K.layers.Activation(activations[i]))

    model.compile(loss = 'mse', optimizer = optimizer, metrics = ['mae', r2_keras])
    
    print ('\n------------Building NN Model Complete------------------')
    
    model.summary()
    
    return model
    
def train_model(model, X_train, Y_train, X_valid, Y_valid, batch_size, num_epochs):
    
    print ('\n---------------Training/Validating NN Model----------------')
    
    history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = num_epochs, verbose = 0, validation_data = (X_valid, Y_valid), shuffle = True, callbacks = [progress()])
    
    print ('\n-----------Training/Validating NN Model Complete----------------')
    return model, history

class progress(K.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 10 == 0: print('Current Epoch:', epoch)
    
def r2_keras(y_true, y_pred):
    SS_res =  K.backend.sum(K.backend.square(y_true - y_pred)) 
    SS_tot = K.backend.sum(K.backend.square(y_true - K.backend.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.backend.epsilon()) )   

def plot_history(history):
    
    epoch = history.epoch
    train_loss = np.array(history.history['loss'])
    val_loss = np.array(history.history['val_loss'])
    train_mae = np.array(history.history['mean_absolute_error'])
    val_mae = np.array(history.history['val_mean_absolute_error'])
    train_r2 = np.array(history.history['r2_keras'])
    val_r2 = np.array(history.history['val_r2_keras'])
    
    plt.figure(0, figsize=(3.75, 2.75))
    plt.plot(epoch, train_loss, label = 'Train Loss')
    plt.plot(epoch, val_loss, label = 'Valid Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss: Mean Squared Error')
    plt.ylim([0, 500])
    plt.savefig('fig1.png', bbox_inches="tight")

    plt.figure(1, figsize=(3.75, 2.75))
    plt.plot(epoch, train_mae, label = 'Train MAE')
    plt.plot(epoch, val_mae, label = 'Valid MAE')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.savefig('fig2.png', bbox_inches="tight")

    plt.figure(2, figsize=(3.75, 2.75))
    plt.plot(epoch, train_r2, label = 'Train R2')
    plt.plot(epoch, val_r2, label = 'Valid R2')
    plt.legend()
    plt.ylim([0.5, 1])
    plt.xlabel('Epoch')
    plt.ylabel('R2 Coeff')
    plt.savefig('fig3.png', bbox_inches="tight")
    
def plot_test(predictions, Y_test):
    
    plt.figure(3, figsize=(3.75, 2.75))
    plt.plot(Y_test, predictions, 'bo')
    
    plt.axis('equal')
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    plt.gca().set_xlim(left = 0)
    plt.gca().set_ylim(bottom = 0) 
    plt.xlabel('Labels')
    plt.ylabel('Predictions')
    plt.plot([0, 1000], [0, 1000]) 
    plt.savefig('fig4.png', bbox_inches="tight")
    
    plt.figure(4, figsize=(3.75, 2.75))
    error = predictions - Y_test
    plt.hist(error, bins = 10)
    plt.xlabel("Prediction Error")
    plt.ylabel("Count") 
    plt.savefig('fig5.png', bbox_inches="tight")
    
if __name__ == '__main__':
    print("Please use `python run.py --nn` to run this model")
