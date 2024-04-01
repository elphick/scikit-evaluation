"""
Category Feature Analysis
=========================

It is common to model across estimation domains using categorical features.
This example demonstrates how to use the ModelSelection class to compare the performance of the
source model against models fitted independently on the category values.

"""
import logging
from typing import Dict

import pandas as pd
import plotly
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_diabetes
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from elphick.sklearn_viz.model_selection import ModelSelection, metrics
from elphick.sklearn_viz.model_selection.models import Models

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S%z')

# %%
# Load Regression Data
# --------------------
#
# We prepare a `group` variable (a pd.Series) in order to test the performance of modelling independently.

diabetes = load_diabetes(as_frame=True)
x, y = diabetes.data, diabetes.target
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
models_to_test: Dict = Models().fast_regressors()

ms: ModelSelection = ModelSelection(algorithms=models_to_test, datasets=xy, target='progression', pre_processor=pp,
                                    k_folds=10, scorer='r2', group=group,
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

fig = ms.plot(metrics=['moe', 'me'], show_group=True)
fig.update_layout(height=700)
fig

# %%
# Categorical Feature Analysis
# ----------------------------
#
# This analysis will test whether better performance can be achieved by modelling the specified categorical class
# separately, rather than passing it as a feature to the model.

fig = ms.plot_category_analysis(algorithm='LR', dataset=None,
                                metrics=['moe', 'me'])
fig.update_layout(height=700)
plotly.io.show(fig)

# %%
#
# .. admonition:: Info
#
#    We can see that the independent model by group score is marginally greater than 1.0 for the F
#    class but is below 1 for the M class.  However, the error margins are very high, so while the
#    test indicates an indepndent model for F would be better that cannot be proven.