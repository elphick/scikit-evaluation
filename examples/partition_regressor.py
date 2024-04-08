"""
Partition Estimator
===================

There are times when the modeller or subject-matter expert feels the need to test estimation domains (data partitions).
The partitions are defined by setting a criteria string per estimation domain.  The criteria string is used to
filter the incoming feature dataframe before fitting the model for that partition.

The idea supporting partitioning is that each partition, having a specific structure can be fitted better.
The trade-off of course is that when partitioned, less data is available for fitting.

"""

import pandas as pd
import plotly
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from elphick.sklearn_viz.components.estimators import PartitionRegressor
from elphick.sklearn_viz.features import plot_feature_importance
from elphick.sklearn_viz.model_selection import ModelSelection

# %%
# Load Regression Data
# --------------------
#
# The California housing dataset will be loaded to demonstrate the regression.

x, y = fetch_california_housing(return_X_y=True, as_frame=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
xy_train: pd.DataFrame = pd.concat([x_train, y_train], axis=1)

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
    ], verbose_feature_names_out=False
)

pp: Pipeline = make_pipeline(preprocessor).set_output(transform='pandas')
# %%
# Define the Estimators
# ---------------------

# %%
# Baseline Model
# ~~~~~~~~~~~~~~
#
# The baseline model will simply be fitted as normal - no partitions will be applied.

base_mdl: Pipeline = make_pipeline(pp, LinearRegression())
base_mdl

# %%
# Partitioned Model
# ~~~~~~~~~~~~~~~~~
#
# We will create the criteria for 3 arbitrary partitions of room size at the lower and upper quartile.
# We'd like to work in the incoming feature space, but the PartitionRegressor will need the criteria in the
# post-processed space, since that is the only data it sees.

x_train['AveRooms'].describe().T

# %%

partition_criteria: dict = {'small': 'AveRooms < 4.4',
                            'medium': '(AveRooms >= 4.4) and (AveRooms < 6.0)',
                            'large': 'AveRooms >= 6.0'}
# %%
# For now this is conversion must be done by the user, manually
pp.fit_transform(x_train)['AveRooms'].describe()

# %%
partition_criteria: dict = {'small': 'AveRooms < -0.43',
                            'medium': '(AveRooms >= -0.43) and (AveRooms < 0.24)',
                            'large': 'AveRooms >= 0.24'}

partition_mdl: Pipeline = make_pipeline(pp, PartitionRegressor(LinearRegression(),
                                                               partition_defs=partition_criteria))
partition_mdl

# %%
# In the model visualisation above, expand the arrow next to the partition names (small, medium, large) to
# see the criteria.

# %%
#
# .. tip::
#
#    A trick to avoid transforming the partition criteria values is to embed the preprocessor into every model,
#    rather than having a common preprocessor.  This will create additional computational overhead but is
#    perhaps a nice way of simplifying the workflow.


# %%
# Demo Fit and Predict
# --------------------

base_mdl.fit(X=x_train, y=y_train)
check_is_fitted(base_mdl)
est_base = pd.Series(base_mdl.predict(X=x_test), index=x_test.index, name='base_est')

partition_mdl.fit(X=x_train, y=y_train)
check_is_fitted(partition_mdl)
est_partition = partition_mdl.predict(X=x_test)

est: pd.DataFrame = pd.concat([est_base, est_partition], axis=1)
est.columns = ['base_est', 'partition_est']
est.head()

# %%
# Cross Validation
# ----------------
ms: ModelSelection = ModelSelection(algorithms={'base-model': base_mdl,
                                                'partition-model': partition_mdl},
                                    datasets=xy_train,
                                    target='MedHouseVal',
                                    group=partition_mdl[-1].domains_, random_state=1234)
fig = ms.plot(show_group=True, metrics=['r2_score'])
fig.update_layout(height=600)
plotly.io.show(fig)

# %%
# To some extent the error margin (notch width) will be driven by the number of samples.
# Let's check the sample counts.

partition_mdl[-1].domains_.value_counts()

# %%
#
# .. note::
#
#    1. In this case the partitioning caused the model performance to deteriorate.
#    2. It appears that (for the small class) the error margins are wider for the partitioned model,
#       likely caused by lower sample count in the fitted model.

# %%
# Feature Importance
# ------------------
#
# We can check that the feature imports works as expected on our new Estimator.

fig = plot_feature_importance(partition_mdl, permute=True, x_test=x_train, y_test=y_train)
fig
