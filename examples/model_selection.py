"""
===============
Model Selection
===============

This example demonstrates a model selection plot incorporating cross validation and test error.

Code has been adapted from the
`machinelearningmastery example <https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/>`_

"""
import logging
from typing import Dict

import pandas
import pandas as pd
import plotly
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler

from elphick.sklearn_viz.model_selection import ModelSelection, plot_model_selection, metrics
from elphick.sklearn_viz.model_selection.models import Models

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S%z')

# %%
# Load Data
# ---------
#
# Once loaded we'll create the train-test split for a classification problem.

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
x = pd.DataFrame(array[:, 0:8], columns=names[0:8])
y = pd.Series(array[:, 8], name=names[8])
xy: pd.DataFrame = pd.concat([x, y], axis=1)

# %%
# Instantiate
# -----------
#
# Create an optional pre-processor as a sklearn Pipeline.

pp: Pipeline = make_pipeline(StandardScaler())
models_to_test: Dict = Models().fast_classifiers()
pp

# %%
# Plot using the function
# -----------------------

fig = plot_model_selection(algorithms=models_to_test, datasets=xy, target='class', pre_processor=pp)
fig.update_layout(height=600)
fig

# %%
# Plot using the object
# ---------------------
#
# We pass in the test data for additional context, and calculate across 30 folds.
# The test data score is plotted as an orange marker.

ms: ModelSelection = ModelSelection(algorithms=models_to_test, datasets=xy, target='class', pre_processor=pp,
                                    k_folds=30)
fig = ms.plot(title='Model Selection', metrics='f1')
fig.update_layout(height=600)
# noinspection PyTypeChecker
plotly.io.show(fig)  # this call to show will set the thumbnail for use in the gallery

# %%
# View the data

ms.data

# %%
# Regressor Model Selection
# -------------------------
#
# Testing different Algorithms

diabetes = load_diabetes(as_frame=True)
x, y = diabetes.data, diabetes.target
y.name = "progression"
xy: pd.DataFrame = pd.concat([x, y], axis=1)
group: pd.Series = pd.Series(x['sex'] > 0, name='grp_sex', index=x.index)

pp: Pipeline = make_pipeline(StandardScaler())
models_to_test: Dict = Models().fast_regressors()

ms: ModelSelection = ModelSelection(algorithms=models_to_test, datasets=xy, target='progression', pre_processor=pp,
                                    k_folds=5, scorer='r2')
fig = ms.plot(metrics={'moe': metrics.moe_95, 'me': metrics.mean_error},
              color_group=group)
fig.update_layout(height=600)
fig

# %%
# Comparing Datasets
# ------------------
#
# Next we will demonstrate a single Algorithm with multiple datasets.

datasets: Dict = {'DS1': xy, 'DS2': xy.sample(frac=0.4)}

fig = plot_model_selection(algorithms=LinearRegression(), datasets=datasets, target='progression', pre_processor=pp,
                           k_folds=5)
fig.update_layout(height=600)
fig
