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
    
    save_path = 'model-and-weights.h5'
    
    # initialize X, Y
    x_train = train_set
    x_valid = valid_set
    x_test = test_set
    
    y_train = train_labels
    y_valid = valid_labels
    y_test = test_labels
    
    # save_data('data.h5', [x_train, y_train, x_valid, y_valid, x_test, y_test])

    # convert to np and slice X, Y
    x_train, y_train, x_valid, y_valid, x_test, y_test = convert2np(x_train, y_train, x_valid, y_valid, x_test, y_test)
    
    # shuffle
    x_train, y_train = shuffle(x_train, y_train)
    x_valid, y_valid = shuffle(x_valid, y_valid)
    
    # normalize features
    x_train, x_valid, x_test = norm_features(x_train, x_valid, x_test)
    
    # build model
    n_train = x_train.shape[1]
    model = build_model(layer_dims, activations, dropout, n_train, lr)
        
    # train model
    model, history = train_model(model, x_train, y_train, x_valid, y_valid, batch_size, num_epochs)
    model.save(save_path)
    
    # plot train/valid loss (outputs graphs of MSE, MAE, and R2 Coeff.)
    print_stats(history)
    plot_history(history)
    
    # predict using model
    predictions = model.predict(x_test, verbose=0)
        
    # plot test results (outputs graphs of labels vs. predictions and histogram of prediction error)
    plot_test(predictions, y_test)


def convert2np(x_train, y_train, x_valid, y_valid, x_test, y_test):
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_valid, y_valid = np.array(x_valid), np.array(y_valid)
    x_test, y_test = np.array(x_test), np.array(y_test)
    
    x_train = x_train[:, :]
    x_valid = x_valid[:, :]
    x_test = x_test[:, :]
    
    y_train = np.reshape(y_train[:, 0], (y_train.shape[0], 1))
    y_valid = np.reshape(y_valid[:, 0], (y_valid.shape[0], 1))
    y_test = np.reshape(y_test[:, 0], (y_test.shape[0], 1))
    
    print('\n---------------NN Parameters----------------')
    print('Train features shape:', x_train.shape)
    print('Train labels shape:', y_train.shape)
    print('Valid features shape:', x_valid.shape)
    print('Valid labels shape:', y_valid.shape)
    print('Test features shape:', x_test.shape)
    print('Test labels shape:', y_test.shape)
    print('--------------------------------------------\n')
    
    return x_train, y_train, x_valid, y_valid, x_test, y_test 


def shuffle(x, y):
    
    order = np.argsort(np.random.random(x.shape[0]))
    x = x[order, :]
    y = y[order, :]
    
    return x, y


def norm_features(x_train, x_valid, x_test):
    
    print('\n---------------Normalizing Features----------------')
    
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    
    x_train = (x_train - mean) / std
    x_valid = (x_valid - mean) / std
    x_test = (x_test - mean) / std

    print('x_train, x_valid, x_test normalized to zero mean and STD 1.\n')

    return x_train, x_valid, x_test  


def build_model(layer_dims, activations, dropout, n_train, lr):
    
    print('\n---------------Building NN Model----------------')
    
    tf.reset_default_graph()
    K.backend.clear_session()
    
    model = K.models.Sequential()
    
    model.add(K.layers.Dense(layer_dims[0], input_dim=n_train))
    model.add(K.layers.Activation(activations[0]))
    for i in range(1, len(layer_dims)):
        model.add(K.layers.Dense(layer_dims[i]))
        model.add(K.layers.Dropout(dropout[i]))
        model.add(K.layers.Activation(activations[i]))

#    model.compile(loss = 'mse', optimizer = K.optimizers.Adam(lr), metrics = ['mae'])
    model.compile(loss='mse', optimizer=K.optimizers.Adam(lr), metrics=['mae', r2_keras])
    
    print('\n------------Building NN Model Complete------------------')
    
    model.summary()
    
    return model


def train_model(model, x_train, y_train, x_valid, y_valid, batch_size, num_epochs):
    
    print('\n---------------Training/Validating NN Model----------------')
    
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=0,
                        validation_data=(x_valid, y_valid), shuffle=True, callbacks=[Progress()])
    
    print('\n-----------Training/Validating NN Model Complete----------------')
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


class Progress(K.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 10 == 0:
            print('Current Epoch:', epoch)


def r2_keras(y_true, y_pred):
    ss_res = K.backend.sum(K.backend.square(y_true - y_pred))
    ss_tot = K.backend.sum(K.backend.square(y_true - K.backend.mean(y_true))) 
    return 1 - ss_res/(ss_tot + K.backend.epsilon())


if __name__ == '__main__':
    print("Please use `python run.py --nn` to run this model")
