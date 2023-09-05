"""
==============
Plot Residuals
==============

This example demonstrates plotting errors / residuals.

Code has been adapted from the
`plotly example <https://plotly.com/python/ml-regression/>`_

"""
import numpy as np
import pandas as pd
import plotly
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets, linear_model

from elphick.sklearn_viz.features import plot_feature_importance, FeatureImportance
import logging

from elphick.sklearn_viz.residuals.error import Errors

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S%z')

# %%
# Data generation and model fitting
# ---------------------------------
# We mimic a simple model fit per the sklearn example.


# Load the diabetes dataset
# diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
diabetes = datasets.load_diabetes(as_frame=True)
X, y = diabetes.data, diabetes.target
y.name = "progression"

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=13
)

mdl = make_pipeline(Lasso())
mdl.set_output(transform="pandas")
mdl.fit(X_train, y_train)

# %%
# Demonstrate the plot
# --------------------


obj_res: Errors = Errors(mdl=mdl, x_test=X_test, y_test=y_test)
fig = obj_res.plot()
# noinspection PyTypeChecker
plotly.io.show(fig)  # this call to show will set the thumbnail for use in the gallery




