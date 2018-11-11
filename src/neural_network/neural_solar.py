import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow import keras as K

# https://medium.com/@rajatgupta310198/getting-started-with-neural-network-for-regression-and-tensorflow-58ad3bd75223
# https://www.kaggle.com/zoupet/neural-network-model-for-house-prices-tensorflow
# https://www.tensorflow.org/tutorials/keras/basic_regression

def main(train_set, valid_set, test_set):
    print("Running NN main")

    print(f"Found {len(train_set.X)} examples with {len(train_set.labels)} labels.")

    train_data = train_set.X
    train_labels = train_set.labels
    train_area = train_set.get_area_labels()
    train_tiles = train_set.get_tile_count_labels()
    train_system = train_set.get_system_count_labels()

    # See above for how to access the data and labels
    
    np.random.seed(1)
    x = np.random.random((100, 20)) * 100
    y = np.random.random((100, 1)) * 100
    
    layer_dims = np.array([32, 8, 1])
    act_funcs = np.array(['relu', 'relu', 'relu'])
    train_epochs = 150
    
    NN_train(x, y, layer_dims, act_funcs, train_epochs)
    NN_predict()
    
def NN_train(train_data, train_labels, layer_dims, act_funcs, train_epochs):
          
    # num features
    n = train_data.shape[1]
    
    # shuffle training set
    ft_shuffled, lb_shuffled = shuffle(train_data, train_labels)

    # normalize train features
    ft_norm_shuf = normalize_inputs(ft_shuffled)
    
    # build model
    model = build_model(layer_dims, act_funcs, n)
    model.summary()
    
    # train model
    history = model.fit(ft_norm_shuf, lb_shuffled, epochs = train_epochs, verbose = 0)
    
    # plot training step
    plot_train(history.history)    
    
def NN_predict():
    
    None
    
    # predict with model
    # predictions = model.predict(test_data, batch_size = 10, verbose = 0)
    
def shuffle(features, labels):
    
    arg_order = np.random.random(labels.shape[0])
    order = np.argsort(arg_order)
    
    features_shuffled = features[order]
    labels_shuffled = np.reshape(labels[order], (labels.shape[0], 1))
    
    return features_shuffled, labels_shuffled

def normalize_inputs(features):
    
    mean = np.mean(features, axis = 0)
    std = np.std(features, axis = 0)
    features_norm = (features - mean) / std
    
    return features_norm

def build_model(layer_dims, act_funcs, n):
    
    model = K.models.Sequential()
    
    model.add(K.layers.Dense(layer_dims[0], input_shape = (n,)))
    model.add(K.layers.Activation(act_funcs[0]))
    
    for i in range(1, len(layer_dims)):
        model.add(K.layers.Dense(layer_dims[i]))
        model.add(K.layers.Activation(act_funcs[i]))

    model.compile(optimizer='rmsprop', loss='mse')
    
    return model
    
def plot_train(history):
    
    plt.figure(1)
    plt.plot(history['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    

if __name__ == '__main__':
    print("Please use `python run.py --nn` to run this model")
