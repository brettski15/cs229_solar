import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


PLOT_DIR = '../output_plots'


def pca_main(train_set, train_labels, test_set, test_labels):
    print("Scaling zdata")
    sc = StandardScaler()
    train_set.drop(columns=['index'], inplace=True)
    test_set.drop(columns=['index'], inplace=True)
    x_train = sc.fit_transform(train_set)
    x_test = sc.transform(test_set)
    stats = train_labels.describe()
    tile_step_size = math.ceil((stats['tile_count']['max'] - stats['tile_count']['min']) / 7)
    system_step_size = math.ceil((stats['solar_system_count']['max'] - stats['solar_system_count']['min']) / 7)
    area_step_size = math.ceil((stats['total_panel_area']['max'] - stats['total_panel_area']['min']) / 7)
    label_ranges = {
        'tile_count': [],
        'solar_system_count': [],
        'total_panel_area': []
    }
    range_max = 0
    while range_max < stats['tile_count']['max']:
        range_min = range_max
        range_max += tile_step_size
        label_ranges['tile_count'].append({'min': range_min, 'max': range_max})

    range_max = 0
    while range_max < stats['solar_system_count']['max']:
        range_min = range_max
        range_max += system_step_size
        label_ranges['solar_system_count'].append({'min': range_min, 'max': range_max})
    range_max = 0

    while range_max < stats['total_panel_area']['max']:
        range_min = range_max
        range_max += area_step_size
        label_ranges['total_panel_area'].append({'min': range_min, 'max': range_max})
    # data_scaled = pd.DataFrame(preprocessing.scale(train_set), columns=train_set.columns)

    print("Applying PCA")
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_train)
    # x_test = pca.transform(x_test)
    # pca_vals = pca.explained_variance_ratio_
    # print(pca_vals)
    # print(pca.components_)
    # pca_df = pd.DataFrame(pca.components_, columns=train_set.columns, index=['PC-1', 'PC-2'])
    # print(x_pca)

    colors = ['b', 'g', 'r', 'm', 'k', 'c', 'y']
    tile_labels = np.array(train_labels['tile_count'])
    system_labels = np.array(train_labels['solar_system_count'])
    area_labels = np.array(train_labels['total_panel_area'])
    np_labels = [tile_labels, system_labels, area_labels]

    for l, ranges in zip(np_labels, label_ranges.keys()):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
        ax.set_title(f'2 component PCA for {ranges.capitalize()}', fontsize=20)
        for c, r in zip(colors, label_ranges[ranges]):
            ax.scatter(x_pca[(tile_labels > r['min']) & (tile_labels < r['max']), 0],
                       x_pca[(tile_labels > r['min']) & (tile_labels < r['max']), 1],
                       color=c, lw=2, label=f'Between {r["min"]} and {r["max"]}')
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.savefig(f'{PLOT_DIR}/2_pca_{ranges}.png')

if __name__ == '__main__':
    print("Please use `python run.py --pca` to run this model")
