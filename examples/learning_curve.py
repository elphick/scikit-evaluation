"""
==============
Learning Curve
==============

This example demonstrates the learning curve, which helps answer two questions:

1) Is my model over-fitted?
2) Will my model benefit from more data?

Code has been adapted from the
`sklearn example <https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py>`_.

This `machinelearningmastery article <https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/>`_ is a great resource for interpretation of learning curves.

"""
import logging

import plotly
from sklearn.datasets import load_digits, load_diabetes
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler

from elphick.sklearn_viz.model_selection import LearningCurve, plot_learning_curve, metrics

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S%z')

# %%
# Load Data
# ---------

X, y = load_digits(return_X_y=True)

# %%
# Create a Classifier Pipeline
# ----------------------------
#
# The pipeline will likely include some pre-processing.

pipe: Pipeline = make_pipeline(StandardScaler(), GaussianNB()).set_output(transform='pandas')
pipe

# %%
# Plot using the function
# -----------------------

cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)
fig = plot_learning_curve(pipe, x=X, y=y, cv=cv)
fig.update_layout(height=600)
# noinspection PyTypeChecker
plotly.io.show(fig)

# %%
# Plot using the object
# ---------------------
#
# Plotting using the object allows access to the underlying data.
#
# .. tip::
#    You can use n_jobs to parallelize the computation.

lc: LearningCurve = LearningCurve(pipe, x=X, y=y, cv=5, n_jobs=5)
fig = lc.plot(title='Learning Curve').update_layout(height=600)
fig

# %%
# View the data

lc.results

# %%
# Results as a dataframe

df = lc.results.get_results()
df.head(10)

# %%
# Regressor Learning Curve
# ------------------------
#
# This example uses a regression model.

diabetes = load_diabetes(as_frame=True)
X, y = diabetes.data, diabetes.target
y.name = "progression"

pipe: Pipeline = make_pipeline(StandardScaler(), RidgeCV()).set_output(transform='pandas')
pipe

# %%
lc: LearningCurve = LearningCurve(pipe, x=X, y=y, cv=5)
fig = lc.plot(title='Learning Curve').update_layout(height=600)
fig

# %%
# Learning Curve with Metrics
# ---------------------------
#
# While a model is fitted based on the defined scorer, we may be interested in other metrics.
# The `metrics` parameter allows us to define additional metrics to calculate.

lc: LearningCurve = LearningCurve(pipe, x=X, y=y,
                                  metrics={'mse': metrics.mean_squared_error, 'moe': metrics.moe_95},
                                  cv=5)
fig = lc.plot(title='Learning Curve with Metrics', metrics=['mse', 'moe'], col_wrap=2).update_layout(height=800)
fig

# %%
# Learning Curve for a metric without the scorer
fig = lc.plot(title='Learning Curve - Metric, no scorer', metrics=['moe'], plot_scorer=False).update_layout(height=700)
fig
