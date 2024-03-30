"""
Compare Multiple N-D datasets
=============================

Comparing 1D datasets readily achieved by overlaying distributions, boxplots, etc.
For multivariate (N-D) datasets, things get a little more difficult.

This example applies Principal Component Analysis by group variable and colors the loading vectors by group.

"""
import logging

import pandas as pd
import plotly.io

from elphick.sklearn_viz.features import plot_parallel_coordinates
from elphick.sklearn_viz.features.principal_components import plot_loading_vectors, plot_correlation_circle, \
    plot_explained_variance, plot_principal_components

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S%z')
# %%
# Create a dataset
# ----------------


import numpy as np

# for consistent results
np.random.seed(7)

n_samples = 125
n_outliers = 25
n_features = 4

# generate Gaussian data of shape (125, 4)
cov1 = np.array([[9, -7, -2, -2],
                 [-7, 7, 1.5, 1],
                 [-2, 1.5, 1, 0.5],
                 [-2, 1, 0.5, 0.5]])
cov2 = np.array([[5, -2, -1.5, -3],
                 [-2, 2, 0.5, 0.5],
                 [-1.5, 0.5, 1, 1],
                 [-3, 0.5, 1, 3]])

x1 = np.dot(np.random.randn(n_samples, n_features), cov1)
x2 = np.dot(np.random.randn(n_samples, n_features), cov2)

df_x1: pd.DataFrame = pd.DataFrame(x1, columns=[f"F{n}" for n in range(1, n_features + 1)])
# shift the mean on two features
df_x1['F4'] = df_x1['F4'] - 2.0
df_x1['F2'] = df_x1['F2'] + 1.0
df_x2: pd.DataFrame = pd.DataFrame(x2, columns=[f"F{n}" for n in range(1, n_features + 1)])
x = pd.concat([df_x1.assign(group='one'), df_x2.assign(group='two')], axis=0).reset_index(drop=True)
x['group'] = pd.Categorical(x['group'])
x

# %%
# Explore the data
# ----------------
#
# The parallel coordinate plot is a good place to start.  The differences in mean and variance across some
# features is clear.

fig = plot_parallel_coordinates(x, color='group')
fig

# %%
# Explore the Principal Components
# --------------------------------
#
# Note that in the next plot, the data is colored by group, but the loading vectors are for the entire dataset.

fig = plot_principal_components(plot_3d=False, x=x.drop(columns=['group']), color=x['group'])
fig.update_layout(height=800)
fig

# %%
# This example below allows the loading vectors to be visualised by each group without the data for clarity.
fig = plot_loading_vectors(x=x.drop(columns=['group']), color=x['group'])
fig

# %%
# And finally, by standardising the input data prior to PCA analysis allows the correlation circle to be shown by group.
fig = plot_correlation_circle(x=x.drop(columns=['group']), color=x['group'])
fig.update_layout(height=800, width=800)
plotly.io.show(fig)


