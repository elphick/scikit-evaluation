"""
Category Feature Analysis
=========================

It is common to model across estimation domains using categorical features.
This example demonstrates how to use the ModelSelection class to compare the performance of the
source model against models fitted independently on the category values.

"""
import logging
from functools import partial
from typing import Dict

import pandas as pd
import plotly
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from elphick.sklearn_viz.model_selection import ModelSelection, metrics

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S%z')

# %%
# Load Regression Data
# --------------------
#
# We prepare a `group` variable (a pd.Series) in order to test the performance of modelling independently.

diabetes = load_diabetes(as_frame=True)
x, y = diabetes.data.copy(), diabetes.target
x['sex'] = pd.Categorical(x['sex'].apply(lambda x: 'M' if x < 0 else 'F'))  # assumed mock classes.
y.name = "progression"
xy: pd.DataFrame = pd.concat([x, y], axis=1)
group: pd.Series = x['sex']

# %%
# Define the pipeline
# -------------------

numerical_cols = x.select_dtypes(include=[float]).columns.to_list()
categorical_cols = x.select_dtypes(include=[object, 'category']).columns.to_list()

categorical_preprocessor = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
numerical_preprocessor = StandardScaler()
preprocessor = ColumnTransformer(
    [
        ("one-hot-encoder", categorical_preprocessor, categorical_cols),
        ("standard_scaler", numerical_preprocessor, numerical_cols),
    ]
)

pp: Pipeline = make_pipeline(preprocessor)
models_to_test: Dict = {'LR': sklearn.linear_model.LinearRegression(),
                        'LASSO': sklearn.linear_model.LassoCV()}

ms: ModelSelection = ModelSelection(algorithms=models_to_test, datasets=xy, target='progression', pre_processor=pp,
                                    k_folds=10, scorer='r2', group=group,
                                    metrics={'r2_score': metrics.r2_score, 'moe': metrics.moe_95,
                                             'rmse': partial(mean_squared_error, squared=False),
                                             'me': metrics.mean_error},
                                    random_state=123)
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

fig = ms.plot(metrics=['moe', 'me'], show_group=True)
fig.update_layout(height=700)
fig

# %%
# Categorical Feature Analysis
# ----------------------------
#
# This analysis will test whether better performance can be achieved by modelling the specified categorical class
# separately, rather than passing it as a feature to the model.

fig = ms.plot_category_analysis(algorithm='LR')
fig.update_layout(height=700)
plotly.io.show(fig)

# %%
# We can view more metrics...

fig = ms.plot_category_analysis(algorithm='LR', dataset=None,
                                metrics=['r2_score', 'moe', 'rmse', 'me'],
                                col_wrap=2)
fig.update_layout(height=800)
fig

# %%
#
# .. admonition:: Info
#
#    We can see from the notch positions of the comparative boxplots, that modelling by group would offer no benefit
#    for either the F or M classes.
