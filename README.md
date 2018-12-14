# cs229_solar

Setup:
------
1. Make a Python3 virtual environment for use with this project
2. Run `pip install -r cs229_solar/requirements.txt`

In order to run this submission code, please `cd` into
the cs229_solar/src directory. From there, execute:
```bash
python run.py
```
with a selection of options from below. Please note that `-c 0` is
necessary to force running the code on the entire data set for any
of the options below. Without this flag, the code will default to
only 1000 examples in a given run. Any number other than 0 will be
used as the number of rows to include in the run. From this set, any
rows with missing values will be dropped, and the remaining examples
will be split 80/10/10.

Neural Network:
---------------
```bash
python run.py --nn -c 0
```
Outputs the following figures in the current directory:
- fig1.png -- Loss : Mean Square Error Over Training Epoch (Train and Dev)
- fig2.png -- Mean Absolute Error Over Training Epoch (Train and Dev)
- fig3.png -- R<sup>2</sup> Correlation Over Training Epoch (Train and Dev)
- fig4.png -- Predictions vs Labels (Test)
- fig5.png -- Histogram of Prediction Error (Test)


Support Vector Machine:
-----------------------
```bash
python run.py --svm -c 0 --kern [rbf, linear, poly]
```
Outputs a scatter plot of the labels versus predictions with the
R<sup>2</sup> correlation coefficient in the title. The default kernel is rbf.

Primary Component Analysis:
---------------------------
```bash
python run.py --pca -c 0
```
Prints out PCA variance contributors, once sorted by PC-1, once by PC-2.
Also outputs a PCA plot of each of the 3 possible labels to:
- cs229_solar/output_plots/2_pca_solar_system_count.png
- cs229_solar/output_plots/2_pca_tile_count.png
- cs229_solar/output_plots/2_pca_total_panel_area.png

Choropleths (Heatmaps) of data:
--------------------------------
```bash
python run.py --heatmap
```
This was used to generate some of the charts included in the
poster and the write-up. Requires a plotly account.