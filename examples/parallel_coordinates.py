"""
====================
Parallel Coordinates
====================

Parallel coordinate plots are very useful for Exploratory Data Analysis (EDA).

Typically the target variable will be colored, since it is the variable of most interest, though this is optional.

The interactive nature of plotly is a real asset for this particular plot. Records/samples can be highlighted by
clicking and dragging the mouse vertically at a given axis for a variable (feature or target).  Multiple selections
are possible.  Single clicking a selection will remove it.

"""

import pandas as pd
import plotly.io as pio
from sklearn.datasets import load_diabetes, load_wine

from elphick.sklearn_viz.features.parallel_coordinates import plot_parallel_coordinates

# %%
# Load Classification Data
# ------------------------

wine = load_wine(as_frame=True)
X, y = wine.data, wine.target.rename('target')
df = pd.concat([X, y], axis=1)
df

# %%
# Plot Classification Data
# ------------------------

fig = plot_parallel_coordinates(pd.concat([X, y], axis=1), target=y.name)
# noinspection PyTypeChecker
pio.show(fig)

# %%
# The target is optional.  If the plot is too dense, then consider sampling as demonstrated.

fig = plot_parallel_coordinates(df.sample(frac=0.5))
fig

# %%
# Load Regression Data
# --------------------

diabetes = load_diabetes(as_frame=True, scaled=False)
X, y = diabetes.data, diabetes.target.rename('target')
df = pd.concat([X, y], axis=1)
df

# %%
# Plot Regression Data
# --------------------

fig = plot_parallel_coordinates(df, target=y.name)
fig

# %%
# Categorical data is supported

df['sex'] = df['sex'].map({1: 'Male', 2: 'Female'}).astype('category')
fig = plot_parallel_coordinates(df.sample(frac=0.5), target=y.name)
fig
