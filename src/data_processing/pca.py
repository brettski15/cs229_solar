import math
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


PLOT_DIR = '../output_plots'


def pca_main(train_set, train_labels, test_set, test_labels):
    print("Scaling data")
    sc = StandardScaler()
    # train_set.drop(columns=['index'], inplace=True)
    # test_set.drop(columns=['index'], inplace=True)
    x_train = sc.fit_transform(train_set)
    x_test = sc.transform(test_set)

    print("Applying PCA")
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_train)
    # x_test = pca.transform(x_test)
    # pca_vals = pca.explained_variance_ratio_
    # print(pca_vals)
    # print(pca.components_)
    # pca_df = pd.DataFrame(pca.components_, columns=train_set.columns, index=['PC-1', 'PC-2'])
    # print(x_pca)

    stats = train_labels.describe()
    label_ranges = {}

    num_colors = 30
    # colors = ['b', 'g', 'r', 'm', 'k', 'c', 'y']
    cmap = plt.cm.get_cmap('Paired', num_colors)
    colors = []
    for i in range(num_colors):
        colors.append(cmap(i))
    random.shuffle(colors)

    for l in train_labels.columns:
        print(f"Plotting PCA analysis for {l}")
        step = math.ceil((stats[l]['max'] - stats[l]['min']) / num_colors)
        range_max = 0
        label_ranges[l] = []
        while range_max < stats[l]['max']:
            range_min = range_max
            range_max += step
            label_ranges[l].append({'min': range_min, 'max': range_max})
        np_labels = np.array(train_labels[l])

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
        ax.set_title(f'2 component PCA for {l.capitalize()}', fontsize=20)
        for c, r in zip(colors, label_ranges[l]):
            ax.scatter(x_pca[(np_labels > r['min']) & (np_labels < r['max']), 0],
                       x_pca[(np_labels > r['min']) & (np_labels < r['max']), 1],
                       color=c, lw=2, label=f'Between {r["min"]} and {r["max"]}')
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.savefig(f'{PLOT_DIR}/2_pca_{l}.png')

if __name__ == '__main__':
    print("Please use `python run.py --pca` to run this model")
