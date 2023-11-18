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

import numpy as np
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

np.random.seed(1234)
pp: Pipeline = make_pipeline(StandardScaler())
models_to_test: Dict = Models().fast_classifiers()
pp

# %%
# Plot using the function
# -----------------------
#
# The box colors are scaled to provide a relative indication of performance based on the score (Kudos to
# `Shah Newaz Khan <https://towardsdatascience.com/applying-a-custom-colormap-with-plotly-boxplots-5d3acf59e193>`_)

fig = plot_model_selection(algorithms=models_to_test, datasets=xy, target='class', pre_processor=pp)
fig.update_layout(height=600)
fig

# %%
# Plot using the object
# ---------------------
#
# The alternative to using the function is to instantiate a ModelSelection object.  This has the advantage of
# persisting the data, which provides greater flexibility and faster re-plotting.
# If metrics as provided additional subplots are provided - however since metrics have no concept of "greater-is-good"
# like a scorer, they are not coloured.

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
# Of course we're not limited to classification problems.  We will demonstrate a regression problem, with multiple
# metrics.  We prepare a `group` variable (a pd.Series) in order to calculate the metrics by group for each fold.

diabetes = load_diabetes(as_frame=True)
x, y = diabetes.data, diabetes.target
y.name = "progression"
xy: pd.DataFrame = pd.concat([x, y], axis=1)
group: pd.Series = pd.Series(x['sex'] > 0, name='grp_sex', index=x.index)

pp: Pipeline = make_pipeline(StandardScaler())
models_to_test: Dict = Models().fast_regressors()

ms: ModelSelection = ModelSelection(algorithms=models_to_test, datasets=xy, target='progression', pre_processor=pp,
                                    k_folds=30, scorer='r2', group=group,
                                    metrics={'moe': metrics.moe_95, 'me': metrics.mean_error})
# %%
# Next we'll view the plot, but we will not (yet) leverage the group variable.

fig = ms.plot(metrics=['moe', 'me'])
fig.update_layout(height=700)
fig

# %%
# Now, we will re-plot using group.  This is fast, since the fitting metrics were calculated when the first plot was
# created, and do not need to be calculated again.
#
# Plotting by group can (hopefully) provide evidence that metrics are consistent across groups.

fig = ms.plot(metrics=['moe', 'me'], show_group=True, col_wrap=2)
fig.update_layout(height=700)
fig

# %%
# Clearly, plot real estate will become a problem for more than 2 or 3 classes - here we used col_wrap mitigate that.

# %%
# Comparing Datasets
# ------------------
#
# Next we will demonstrate a single Algorithm with multiple datasets.  This is useful when exploring features that
# improve model performance.

datasets: Dict = {'DS1': xy, 'DS2': xy.sample(frac=0.4)}

fig = plot_model_selection(algorithms=LinearRegression(), datasets=datasets, target='progression', pre_processor=pp,
                           k_folds=30)
fig.update_layout(height=600)
fig
