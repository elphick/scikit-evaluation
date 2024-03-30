"""
==============================
Multivariate Outlier Detection
==============================

Mahalanobis Distance can be used to detect outliers in multivariate space.
This can be combined with Principal Component Analysis (PCA) to reduce dimensionality prior to outlier detection.

"""
import logging

import pandas as pd
import plotly.io as pio
import plotly.express as px
from sklearn.datasets import load_diabetes

from elphick.sklearn_viz.features import plot_principal_components, plot_scatter_matrix, \
    plot_explained_variance, PrincipalComponents, plot_parallel_coordinates
from elphick.sklearn_viz.features.outlier_detection import OutlierDetection

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S%z')
# %%
# Create a dataset
# ----------------
#
# From the `sklearn example <https://scikit-learn.org/stable/auto_examples/covariance/plot_mahalanobis_distances.html>`_

import numpy as np

# for consistent results
np.random.seed(7)

n_samples = 125
n_outliers = 25
n_features = 4

# generate Gaussian data of shape (125, 4)
gen_cov = np.eye(n_features)
gen_cov[0, 0] = 2.0
X = np.dot(np.random.randn(n_samples, n_features), gen_cov)
# add some outliers
outliers_cov = np.eye(n_features)
outliers_cov[np.arange(1, n_features), np.arange(1, n_features)] = 7.0
X[-n_outliers:] = np.dot(np.random.randn(n_outliers, n_features), outliers_cov)

x: pd.DataFrame = pd.DataFrame(X, columns=[f"F{n}" for n in range(1, n_features + 1)])
test_outlier = pd.Series(x['F1'].index > (n_samples - n_outliers - 1), name='test_outlier').astype('category')
x

# %%
# Explore the data and principal components
# -----------------------------------------
#
# The parallel coordinate plot is a good place to start.  The test outlier class variable is used to color the plot.
# Note, the outliers shown are before outlier detection - this is the synthetically generated outlier class.

fig = plot_parallel_coordinates(data=pd.concat([x, test_outlier], axis=1), color='test_outlier')
fig

# %%
# Explore the principal components in 2D and then 3D
fig = plot_explained_variance(x=x)
fig

# %%
fig = plot_principal_components(plot_3d=False, x=x, color=test_outlier)
fig.update_layout(height=800)
fig

# %%
fig = plot_principal_components(plot_3d=True, x=x, color=test_outlier)
fig.update_layout(height=800)
fig

# %%
# Detect Outliers
# ---------------

od: OutlierDetection = OutlierDetection(x=x, pca_spec=2)
detected_outliers: pd.Series = od.data['outlier']

# %%
# Visualise the detected outliers with the default p_val of 0.001

fig = plot_principal_components(plot_3d=False, x=x, color=detected_outliers)
fig.update_layout(height=800)
fig

# %%
# We can tighten the threshold to align more closely with our expectation.

detected_outliers: pd.Series = OutlierDetection(x=x, pca_spec=2, p_val=0.25).data['outlier']
fig = plot_principal_components(plot_3d=False, x=x, color=detected_outliers)
fig.update_layout(height=800)
# noinspection PyTypeChecker
pio.show(fig)

# %%
fig = od.plot_outlier_matrix()
fig.update_layout(height=800)
fig

# %%
# The parallel plot allows us to explore the difference between our defined outliers (test_outliers) and what
# was detected as an outlier (outlier).
fig = plot_parallel_coordinates(data=pd.concat([x, test_outlier, detected_outliers.astype('category')], axis=1),
                                color='outlier')
fig

# %%
# We can detect in the original feature space with pca_spec = 0

detected_outliers: pd.Series = OutlierDetection(x=x, pca_spec=0, p_val=0.25).data['outlier']
detected_outliers.sum()
