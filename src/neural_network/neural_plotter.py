import numpy as np
import matplotlib.pyplot as plt


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
    plt.ylim([0, 2000])
    plt.savefig('../output_plots/NN1-loss.png', bbox_inches="tight")

    plt.figure(1, figsize=(3.75, 2.75))
    plt.plot(epoch, train_mae, label = 'Train MAE')
    plt.plot(epoch, val_mae, label = 'Valid MAE')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.ylim([0, 20])
    plt.savefig('../output_plots/NN2-MAE.png', bbox_inches="tight")

    plt.figure(2, figsize=(3.75, 2.75))
    plt.plot(epoch, train_r2, label = 'Train R2')
    plt.plot(epoch, val_r2, label = 'Valid R2')
    plt.legend()
    plt.ylim([0.6, 1])
    plt.xlabel('Epoch')
    plt.ylabel('R2 Coeff')
    plt.savefig('../output_plots/NN3-R2.png', bbox_inches="tight")
    
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
    plt.plot([0, 3000], [0, 3000]) 
    plt.savefig('../output_plots/NN4-predictions.png', bbox_inches="tight")
    
    plt.figure(4, figsize=(3.75, 2.75))
    error = predictions - Y_test
    error = error[error > -100]
    error = error[error < 100]
    
    plt.hist(error, bins = 10)
    plt.xlabel("Prediction Error")
    plt.ylabel("Count") 
    plt.savefig('../output_plots/NN5-error.png', bbox_inches="tight")