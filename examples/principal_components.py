"""
============================
Principal Component Analysis
============================

Principal Component Analysis is a feature reduction (decomposition) technique that aims to maximise the 
retained variance in less features.  It is a tool to help manage the "curse of dimensionality".

"""
import logging

import pandas as pd
import plotly.io as pio
import plotly.express as px
from sklearn.datasets import load_diabetes

from elphick.sklearn_viz.features import plot_principal_components, plot_scatter_matrix, \
    plot_explained_variance, PrincipalComponents

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S%z')
# %%
# Load Classification Data
# ------------------------

df = px.data.iris().drop(columns=['species_id'])
df['species'] = df['species'].astype('category')
x = df[[col for col in df.columns if col != 'species']]
y = df['species']

# %%
# Plot Classification Data
# ------------------------

# %%
# SPLOM - Original Feature Space
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

fig = plot_scatter_matrix(x=x, y=y, original_features=True)
fig.update_layout(height=800)
fig

# %%
# SPLOM - Principal Components
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

fig = plot_scatter_matrix(x=x, y=y)
fig.update_layout(height=800)
fig

# %%
# Scatter - 2D PCA
# ^^^^^^^^^^^^^^^^

fig = plot_principal_components(x=x, color=y, plot_3d=False, loading_vectors=False)
fig.update_layout(height=800)
fig

# %%
# Plotting loading vectors is the default.
fig = plot_principal_components(x=x, color=y, plot_3d=False)
fig.update_layout(height=800)
# noinspection PyTypeChecker
pio.show(fig)

# %%
# Explained Variance
# ^^^^^^^^^^^^^^^^^^

fig = plot_explained_variance(x=x, y=y)
fig

# %%
# Scatter - 3D PCA
# ^^^^^^^^^^^^^^^^

fig = plot_principal_components(x=x, color=y, loading_vectors=False)
fig.update_layout(height=800)
fig

# %%
# Plotting loading vectors is the default.
fig = plot_principal_components(x=x, color=y)
fig.update_layout(height=800)
fig

# %%
# Regression Datasets
# -------------------
#
# The preceding examples demonstrated a categorical target variable.
# Regression problems with a numeric variable are also supported.

diabetes = load_diabetes(as_frame=True, scaled=False)
x, y = diabetes.data, diabetes.target.rename('target')
df = pd.concat([x, y], axis=1)
df.shape

# %%
fig = plot_principal_components(x=x, color=y, plot_3d=False)
fig.update_layout(height=800)
fig

# %%
# This dataset requires more variables to retain a reasonable proportion of the total variance compared
# to the iris dataset as indicated in the section below.

# %%
# Accessing the Data
# ------------------
#
# By plotting with the object rather than the function you can access the data.

pca = PrincipalComponents(x=x, color=y)
fig = pca.plot_explained_variance()
fig

# %%
pca.data