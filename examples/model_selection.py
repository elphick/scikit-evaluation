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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler

from elphick.sklearn_viz.model_selection import ModelSelection, plot_model_selection
from elphick.sklearn_viz.model_selection.models import Models

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S%z')

# %%
# Load Data
# ---------
#
# Once loaded we'll create the train-test split

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = pd.DataFrame(array[:, 0:8], columns=names[0:8])
y = pd.Series(array[:, 8], name=names[8])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Create pipeline
# ---------------
#
# The pipeline will likely include some pre-processing, typically the simple baseline model.

pipe: Pipeline = make_pipeline(StandardScaler(), LogisticRegression())
pipe.set_output(transform="pandas")
models_to_test: Dict = Models().fast_classifiers()
pipe

# %%
# Plot using the function
# -----------------------

fig = plot_model_selection(pipe, models=models_to_test, x_train=X_train, y_train=y_train, k_folds=5)
fig.update_layout(height=600)
fig

# %%
# Plot using the object
# ---------------------
#
# We pass in the test data for additional context, and calculate across 30 folds.

ms: ModelSelection = ModelSelection(pipe, models=models_to_test, x_train=X_train, y_train=y_train,
                                    k_folds=30, x_test=X_test, y_test=y_test)
fig = ms.plot(title='Model Selection')
fig.update_layout(height=600)
# noinspection PyTypeChecker
plotly.io.show(fig)  # this call to show will set the thumbnail for use in the gallery

# %%
# View the data

ms.data
