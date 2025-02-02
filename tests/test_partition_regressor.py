import pytest
import pandas as pd
from sklearn.base import is_regressor
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from elphick.sklearn_viz.components.estimators import PartitionRegressor

@pytest.fixture
def data():
    x, y = fetch_california_housing(return_X_y=True, as_frame=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test

@pytest.fixture
def preprocessor(data):
    x_train, _, _, _ = data
    numerical_cols = x_train.select_dtypes(include=[float]).columns.to_list()
    categorical_cols = x_train.select_dtypes(include=[object, 'category']).columns.to_list()
    categorical_preprocessor = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    numerical_preprocessor = StandardScaler()
    preprocessor = ColumnTransformer(
        [
            ("one-hot-encoder", categorical_preprocessor, categorical_cols),
            ("standard_scaler", numerical_preprocessor, numerical_cols),
        ], verbose_feature_names_out=False
    )
    return preprocessor

@pytest.fixture
def partition_criteria():
    return {'small': 'AveRooms < -0.43',
            'medium': '(AveRooms >= -0.43) and (AveRooms < 0.24)',
            'large': 'AveRooms >= 0.24'}

def test_partition_regressor_fit(data, preprocessor, partition_criteria):
    x_train, _, y_train, _ = data
    pp = make_pipeline(preprocessor).set_output(transform='pandas')
    partition_mdl = make_pipeline(pp, PartitionRegressor(LinearRegression(), partition_defs=partition_criteria))
    partition_mdl.fit(X=x_train, y=y_train)
    check_is_fitted(partition_mdl[-1])

def test_partition_regressor_predict(data, preprocessor, partition_criteria):
    x_train, x_test, y_train, _ = data
    pp = make_pipeline(preprocessor).set_output(transform='pandas')
    partition_mdl = make_pipeline(pp, PartitionRegressor(LinearRegression(), partition_defs=partition_criteria))
    partition_mdl.fit(X=x_train, y=y_train)
    predictions = partition_mdl.predict(X=x_test)
    assert len(predictions) == len(x_test)

def test_partition_regressor_partitioning(data, preprocessor, partition_criteria):
    x_train, _, y_train, _ = data
    pp = make_pipeline(preprocessor).set_output(transform='pandas')
    partition_mdl = make_pipeline(pp, PartitionRegressor(LinearRegression(), partition_defs=partition_criteria))
    partition_mdl.fit(X=x_train, y=y_train)
    domains = partition_mdl[-1].domains_
    assert set(domains.unique()) == {'small', 'medium', 'large'}

def test_partition_regressor_is_regressor(partition_criteria):
    partition_regressor = PartitionRegressor(LinearRegression(), partition_defs=partition_criteria)
    assert is_regressor(partition_regressor)

