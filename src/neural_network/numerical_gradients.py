import numpy as np
from tensorflow import keras as K

def main():
    
    
    
    
    filename = '../model-and-weights.h5'
    model = K.models.load_model(filename)
    model.summary()
    
    x = np.zeros((1, 171))
    num_grad = np.zeros((1, 171))
    
    epsilon = 1e-6
    for i in range(x.shape[1]):
        x_plus = np.copy(x)
        x_minus = np.copy(x)
        
        x_plus[0, i] = epsilon
        x_minus[0, i] = -epsilon
        
        num_grad[0, i] = (model.predict(x_plus) - model.predict(x_minus)) / (2 * epsilon)
    
    np.savetxt('num_grad.csv', num_grad)

















if __name__ == '__main__':
    main()