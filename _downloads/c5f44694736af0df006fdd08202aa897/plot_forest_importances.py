"""
==========================================
Feature importances with a forest of trees
==========================================

This example shows the use of a forest of trees to evaluate the importance of
features on an artificial classification task. The blue bars are the feature
importances of the forest, along with their inter-trees variability represented
by the error bars.

As expected, the plot suggests that 3 features are informative, while the
remaining are not.

The base code has been adapted from the
`original scikit-learn example <https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#sphx-glr-download-auto-examples-ensemble-plot-forest-importances-py>`_

To learn about the benefits of permuted performance over the importance captured when a model is trained you should
refer to that original example.  This example will focus on the interactive feature importance plot.

"""

import pandas as pd
import plotly
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier

from elphick.sklearn_viz.features import plot_feature_importance, FeatureImportance
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S%z')
# %%
# Data generation and model fitting
# ---------------------------------
# We generate a synthetic dataset with only 3 informative features. We will
# explicitly not shuffle the dataset to ensure that the informative features
# will correspond to the three first columns of X.

X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=3,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    random_state=0,
    shuffle=False,
)
X = pd.DataFrame(X, columns=[f"Feature {i}" for i in range(1, X.shape[1]+1)])
y = pd.Series(y, name='Class')
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# %%
# A random forest classifier will be fitted to compute the feature importances.
#
# .. note ::
#     To obtain the real feature names in the plot the following is needed:
#
#     - Pass pd.DataFrames to the fit method
#     - Set the transform output to "pandas"

pipe = make_pipeline(SelectKBest(k='all'), RandomForestClassifier(random_state=0))
pipe.set_output(transform="pandas")
pipe.fit(X_train, y_train)

# %%
# Feature importance based on mean decrease in impurity
# -----------------------------------------------------
# Long story short, this approach is faster, since it comes as an output of model fitting, but is less accurate.
#
# Create an interactive Feature Importance plot

fig = plot_feature_importance(pipe)
fig

# %% Sort and limit to Top 5

fig = plot_feature_importance(pipe, sort=True, top_k=5)
fig

# %% Plot horizontal, using the object rather than the function

fig = FeatureImportance(pipe).plot(horizontal=True, sort=True, top_k=5)
fig

# %% Show the feature importance data

feature_importance: pd.DataFrame = FeatureImportance(pipe).data
feature_importance

# %%
# As expected, the three first features are found important.
#
# Feature importance based on feature permutation
# -----------------------------------------------
# This approach takes longer but is better.
#
# Create an interactive Feature Importance plot using permutation.

fi = FeatureImportance(pipe, permute=True, x_test=X_test, y_test=y_test)
fig = fi.plot()
fig

# %% Replot without the delay, since the importance data is stored

fig = fi.plot(horizontal=True, sort=True, top_k=5)
# noinspection PyTypeChecker
plotly.io.show(fig)  # this call to show will set the thumbnail for use in the gallery

# %%
# The same features are detected as most important using both methods. Although
# the relative importances vary. As seen on the plots, MDI is less likely than
# permutation importance to fully omit a feature.

# %% Importance of Pipeline input features
# If features are engineered, the reported feature importances (by default) will include the engineered features.

pipe2 = make_pipeline(PolynomialFeatures(degree=2), RandomForestClassifier(random_state=0))
pipe2.set_output(transform="pandas")
pipe2.fit(X_train, y_train)

# %%
fi = FeatureImportance(pipe2, permute=True, x_test=X_test, y_test=y_test)
fig = fi.plot(sort=True, top_k=10)
fig

# %% Sometimes you may want the importance of the (lesser number of) features that are the original pipeline inputs.
# For this we set the pipeline_input_features parameter to True.

fi = FeatureImportance(pipe2, permute=True, x_test=X_test, y_test=y_test, pipeline_input_features=True)
fig = fi.plot()
fig
