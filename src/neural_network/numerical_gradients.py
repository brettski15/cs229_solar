import numpy as np
from tensorflow import keras as K
from .data_processing.parse_csv import get_df_from_csv

def main():
    
    
    
    
    filename = '../model-and-weights.h5'
    model = K.models.load_model(filename)
    model.summary()
    
    x = np.zeros((1, 176))
    num_grad = np.zeros((1, 176))
    
    epsilon = 0.01
    for i in range(x.shape[1]):
        x_plus = np.copy(x)
        x_minus = np.copy(x)
        
        x_plus[0, i] = epsilon
        x_minus[0, i] = -epsilon
        
        num_grad[0, i] = (model.predict(x_plus) - model.predict(x_minus)) / (2 * epsilon)
    
    np.savetxt('num_grad.csv', num_grad)
    
    
    
    tract_all = '../../data/tract_all.csv'
    data, labels = get_df_from_csv(tract_all, 1)
    print(list(data.columns.values))
















if __name__ == '__main__':
    main()